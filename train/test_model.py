import os
import random
from argparse import Namespace
from time import time

import numpy as np
import torch
import oib.models
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from oib.datasets import create_dataset
from oib.external import EXT_PACKAGE
from oib.opt import parse_exp_args, parse_test_args
from oib.utils import builder
from oib.utils.config import get_config
from oib.utils.etqdm import etqdm
from oib.utils.logger import logger
from oib.utils.misc import CONST, TrainMode, bar_perfixes, format_args_cfg
from oib.utils.netutils import clip_gradient, setup_seed
from oib.utils.recorder import Recorder
from oib.utils.summarizer import Summarizer


def _init_fn(worker_id):
    seed = worker_id + int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def main_worker(cfg: CN, arg: Namespace, time_f: float):
    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from oib.external import {pkg}")

    # * we use a single GPU to do the testing.
    # * no rank is needed.
    rank = None

    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    summarizer = Summarizer(recorder.tensorboard_path, rank=rank)

    test_data = create_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET)
    test_loader = DataLoader(
        test_data,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=int(arg.workers),
        drop_last=False,
        collate_fn=test_data.collate_fn,
        worker_init_fn=_init_fn,
    )

    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET, train_cfg=cfg.TRAIN)
    model = DP(model).to(arg.device)

    with torch.no_grad():
        model.eval()
        testbar = etqdm(test_loader, rank=rank)
        for bidx, batch in enumerate(testbar):
            preds, _ = model(batch, "test")
            testbar.set_description(f"{bar_perfixes['test']} Epoch {0} | " f"{model.module.evaluator}")

        recorder.record_evaluator(model.module.evaluator, 0, TrainMode.TEST)
        summarizer.summarize_evaluator(model.module.evaluator, 0, TrainMode.TEST)


if __name__ == "__main__":
    arg, _ = parse_exp_args()
    arg, exp_time = parse_test_args(arg)
    assert (
        arg.n_gpus == 1
    ), """
        We only support a single GPU when testing, or the SCORES WILL BE WRONG!!!
        If the GPU is out of memory, you can try to reduce the batch size.
    """
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    # logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    logger.info("====> Testing on single GPU (Data Parallel) <====")
    main_worker(cfg, arg, exp_time)
