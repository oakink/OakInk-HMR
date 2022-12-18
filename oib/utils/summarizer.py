import os
import time
from typing import Dict, Optional, TypeVar

from torch.utils.tensorboard import SummaryWriter
from oib.metrics.evaluator import Evaluator

from .misc import TrainMode


class DDPSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, rank, **kwargs):
        super(DDPSummaryWriter, self).__init__(log_dir, **kwargs)
        self.rank = rank

    def add_scalar(self, tag, value, global_step=None, walltime=None):
        if self.rank != 0:
            return
        super(DDPSummaryWriter, self).add_scalar(tag, value, global_step=global_step, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.rank != 0:
            return
        return super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"):
        if self.rank != 0:
            return
        return super().add_image(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"):
        if self.rank:
            return
        return super().add_images(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    # there are lots of things can be added to tensorboard, but I don't want to add them all


class Summarizer:
    def __init__(
        self,
        tensorboard_path: str = "./runs",
        rank: Optional[int] = None,
    ) -> None:
        self.rank = rank
        if not self.rank:
            self.tb_writer = SummaryWriter(tensorboard_path)

    def summarize_evaluator(self, evaluator: Evaluator, epoch: int, train_mode: TrainMode):
        if self.rank:
            return
        file_perfixes = {
            TrainMode.TRAIN: "train",
            TrainMode.VAL: "val",
            TrainMode.TEST: "test",
        }
        prefix = file_perfixes[train_mode]
        eval_measures = evaluator.get_measures_all_striped(return_losses=False)
        for k, v in eval_measures.items():
            if isinstance(v, Dict):
                for k_, v_ in v.items():
                    self.tb_writer.add_scalar(f"{k}/{prefix}/{k_}", v_, epoch)
            else:
                self.tb_writer.add_scalar(f"{k}/{prefix}", v, epoch)

        eval_images = evaluator.get_metric_images()

        for k, imgs in eval_images.items():
            for i, img in enumerate(imgs):
                img_key = k + (f"_{i}" if len(imgs) > 1 else "")
                if len(img.shape) == 3:
                    self.tb_writer.add_image(f"{img_key}/{prefix}", img, epoch, dataformats="HWC")
                elif len(img.shape) == 4:
                    gif_img = img.transpose(0, 3, 1, 2)[None]
                    self.tb_writer.add_video(f"{img_key}/{prefix}", gif_img, epoch)
        self.tb_writer.flush()

    def summarize_losses(self, losses: Dict, n_iter: int):
        if self.rank:
            return

        self.tb_writer.add_scalar("Final Loss", losses["final_loss"], n_iter)
        self.tb_writer.add_scalars(
            "Losses", {k: v for k, v in losses.items() if k != "final_loss" and v is not None}, n_iter
        )
        self.tb_writer.flush()

    def summarize_scheduler(self, scheduler):
        if self.rank:
            return
        self.tb_writer.add_scalars("lr", {k: v.get_last_lr()[0] for k, v in scheduler.items()})
        self.tb_writer.flush()
