import os
import pickle
import random
import sys
import time
from pprint import pformat, pprint
from typing import Dict, List, Optional, TypeVar, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import yacs
import yaml
from git import Repo
from PIL.Image import Image
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from oib.metrics.basic_metric import LossMetric
from oib.metrics.evaluator import Evaluator

from .ioutils import load_model, load_random_state, load_train_param, save_states
from .logger import logger
from .misc import RandomState, TrainMode, singleton

T = TypeVar("T", bound="Recorder")


@singleton
class Recorder:
    def __init__(
        self: T,
        exp_id: str,
        cfg: Dict,
        root_path: str = "exp",
        rank: Optional[int] = None,
        time_f: Optional[float] = None,
    ):
        assert (
            exp_id == "default" or exp_id.startswith("debug") or exp_id.startswith("eval") or self.get_git_commit()
        ), "MUST commit before the experiment!"
        self.timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time_f if time_f else time.time()))
        self.exp_id = exp_id
        self.cfg = cfg
        self.dump_path = os.path.join(root_path, f"{exp_id}_{self.timestamp}")
        self.eval_dump_path = os.path.join(self.dump_path, "evaluations")
        self.tensorboard_path = os.path.join(self.dump_path, "runs")
        self.rank = rank
        self._record_init_info()

    def _record_init_info(self: T):
        if not self.rank:
            if not os.path.exists(self.dump_path):
                os.makedirs(self.dump_path)
            if not os.path.exists(self.eval_dump_path):
                os.makedirs(self.eval_dump_path)
            assert logger.filehandler is None, "log file path has been set"
            logger.set_log_file(path=self.dump_path, name=f"{self.exp_id}_{self.timestamp}")
            logger.info(f"run command: {' '.join(sys.argv)}")
            if self.exp_id != "default":
                logger.info(f"git commit: {self.get_git_commit()}")
            with open(os.path.join(self.dump_path, "dump_cfg.yaml"), "w") as f:
                f.write(self.cfg.dump(sort_keys=False))
                # yaml.dump(self.cfg, f, Dumper=yaml.Dumper, sort_keys=False)
            f.close()
            logger.info(f"dump cfg file to {os.path.join(self.dump_path, 'dump_cfg.yaml')}")
        else:
            logger.remove_log_stream()
            logger.disabled = True

    def record_checkpoints(
        self: T,
        model,
        optimizer: Union[Dict[str, Optimizer], Optimizer],
        scheduler: Union[Dict[str, _LRScheduler], _LRScheduler],
        epoch: int,
        snapshot: int,
    ):
        if self.rank:
            return
        checkpoints_path = os.path.join(self.dump_path, "checkpoints")
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        # construct RandomState tuple
        random_state = RandomState(
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            torch_cuda_rng_state_all=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),
            random_rng_state=random.getstate(),
        )

        save_states(
            {
                "epoch": epoch + 1,
                "model": model,
                "optimizer": (
                    optimizer.state_dict()
                    if type(optimizer) is not dict
                    else {k: v.state_dict() for k, v in optimizer.items()}
                ),
                "scheduler": (
                    scheduler.state_dict()
                    if type(scheduler) is not dict
                    else {k: v.state_dict() for k, v in scheduler.items()}
                ),
                "random_state": random_state,
            },
            is_best=False,
            checkpoint=checkpoints_path,
            snapshot=snapshot,
        )
        # logger.info(f"record checkpoints to {checkpoints_path}")

    def resume_checkpoints(
        self: T,
        model,
        optimizer: Union[Dict[str, Optimizer], Optimizer],
        scheduler: Union[Dict[str, _LRScheduler], _LRScheduler],
        resume_path: str,
        resume_epoch: Optional[int] = None,
    ):
        """[summary]

        Args:
            self (T): [description]
            model ([type]): [description]
            optimizer (Union[Dict[str, Optimizer], Optimizer]): [description]
            scheduler (Union[Dict[str, _LRScheduler], _LRScheduler]): [description]
            resume_path (str): [description]
            resume_epoch (Optional[int], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        map_location = f"cuda:{self.rank}" if self.rank is not None else "cuda"
        resume_path = os.path.join(
            resume_path, "checkpoints", f"checkpoint_{resume_epoch}" if resume_epoch else "checkpoint"
        )
        epoch = load_train_param(
            optimizer, scheduler, os.path.join(resume_path, "train_param.pth.tar"), map_location=map_location
        )

        load_random_state(os.path.join(resume_path, "random_state.pkl"))
        load_model(model, resume_path, map_location=map_location)
        return epoch

    # def record_loss(self, loss_metric: LossMetric, epoch: int, comment=""):
    #     if self.rank != 0:
    #         return
    #     loss_dump_path = os.path.join(self.eval_dump_path, f"{comment}Loss.txt")
    #     with open(loss_dump_path, "a") as f:
    #         f.write(f"Epoch {epoch} | {comment} loss metric:\n {pformat(loss_metric.get_measures())}\n\n")

    # def record_metric(self, metrics: List, epoch: int, comment=""):
    #     if self.rank != 0:
    #         return
    #     metric_dump_path = os.path.join(self.eval_dump_path, f"{comment}Metric.txt")
    #     with open(metric_dump_path, "a") as f:
    #         f.write(f"Epoch {epoch} | {comment} metric:\n")
    #         for M in metrics:
    #             f.write(f"{pformat(M.get_measures())}\n")
    #         f.write("\n")

    def record_evaluator(self: T, evaluator: Evaluator, epoch: int, train_mode: TrainMode):
        if self.rank:
            return

        file_perfixes = {
            TrainMode.TRAIN: "train",
            TrainMode.VAL: "val",
            TrainMode.TEST: "test",
        }
        prefix = file_perfixes[train_mode]
        evaluations_path = os.path.join(self.dump_path, "evaluations")
        if not os.path.exists(evaluations_path):
            os.makedirs(evaluations_path)

        images: Dict[str, List[Image]] = evaluator.get_metric_images()
        if len(images):
            images_path = os.path.join(evaluations_path, "images", f"{prefix}_epoch_{epoch}_images")
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            for k, imgs in images.items():
                for i, img in enumerate(imgs):
                    img_path = os.path.join(images_path, f"{k}{f'_{i}' if len(imgs) > 1 else ''}.")
                    if len(img.shape) == 3:  # IMG
                        cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(img_path + "png", cvt_img)
                    elif len(img.shape) == 4:  # GIF
                        with imageio.get_writer(img_path + "gif", mode="I") as w:
                            for frame in img:
                                w.append_data(frame)

        with open(os.path.join(evaluations_path, f"{prefix}_eval.txt"), "a") as f:
            f.write(f"Epoch {epoch} evaluator msg:\n {pformat(evaluator.get_measures_all_striped())}\n\n")

    @staticmethod
    def get_git_commit() -> Optional[str]:
        # get current git report
        repo = Repo(".")

        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        untracked_files = repo.untracked_files

        if len(modified_files):
            logger.error(f"modified_files: {' '.join(modified_files)}")
        if len(staged_files):
            logger.error(f"staged_files: {' '.join(staged_files)}")
        if len(untracked_files):
            logger.error(f"untracked_files: {' '.join(untracked_files)}")

        return (
            repo.head.commit.hexsha if not (len(modified_files) or len(staged_files) or len(untracked_files)) else None
        )
