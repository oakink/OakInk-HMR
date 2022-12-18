import os
import random
from argparse import Namespace
from time import time
import datetime

import numpy as np
import torch
import oib.models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from yacs.config import CfgNode as CN
from oib.datasets import create_dataset
from oib.external import EXT_PACKAGE
from oib.opt import parse_exp_args
from oib.utils import builder
from oib.utils.config import get_config
from oib.utils.etqdm import etqdm
from oib.utils.logger import logger
from oib.utils.misc import CONST, TrainMode, bar_perfixes, format_args_cfg
from oib.utils.netutils import clip_gradient, setup_seed
from oib.utils.recorder import Recorder
from oib.utils.summarizer import Summarizer
from oib.utils.misc import find_free_port


def _init_fn(worker_id):
    seed = worker_id + int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def main_worker(gpu_id: int, cfg: CN, arg: Namespace, time_f: float):

    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from oib.external import {pkg}")

    if arg.distributed:
        rank = arg.n_gpus * arg.node_rank + gpu_id
        torch.distributed.init_process_group(
            arg.dist_backend, rank=rank, world_size=arg.world_size, timeout=datetime.timedelta(seconds=5000)
        )
        assert rank == torch.distributed.get_rank(), "Something wrong with nodes or gpus"
        torch.cuda.set_device(rank)
    else:
        rank = None  # only one process.

    setup_seed(cfg.TRAIN.MANUAL_SEED + rank if rank is not None else cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    summarizer = Summarizer(recorder.tensorboard_path, rank=rank)

    # add a barrier, to make sure all recorders are created
    torch.distributed.barrier()

    train_data = create_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET)
    train_sampler = DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_data,
        batch_size=arg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=int(arg.workers),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        worker_init_fn=_init_fn,
        collate_fn=train_data.collate_fn,
        persistent_workers=True,
    )

    if rank == 0:
        val_data = create_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET)
        val_loader = DataLoader(
            val_data,
            batch_size=arg.val_batch_size,
            shuffle=True,  # ! WARNING: may lead to the validation result is not equal to the testing result
            num_workers=int(arg.workers),
            drop_last=False,
            collate_fn=val_data.collate_fn,
            worker_init_fn=_init_fn,
        )
    else:
        val_loader = None

    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET, train_cfg=cfg.TRAIN)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_GAMMA
    )

    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
    else:
        epoch = 0

    # Make sure model is created, resume is finished
    torch.distributed.barrier()

    logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
    for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
        if arg.distributed:
            train_sampler.set_epoch(epoch_idx)

        model.train()
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()

            preds, loss_dict = model(batch, "train", batch_idx=bidx)
            loss = loss_dict["final_loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()

            summarizer.summarize_losses(loss_dict, epoch_idx * len(train_loader) + bidx)
            trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} | " f"{model.module.evaluator}")

        scheduler.step()
        # logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")
        torch.distributed.barrier()
        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        recorder.record_evaluator(model.module.evaluator, epoch_idx, TrainMode.TRAIN)
        summarizer.summarize_evaluator(model.module.evaluator, epoch_idx, TrainMode.TRAIN)
        model.module.evaluator.reset_all()

        if arg.eval_interval != -1 and epoch_idx % arg.eval_interval == arg.eval_interval - 1 and rank == 0:
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=rank)
                for bidx, batch in enumerate(valbar):
                    batch = model._recursive_to(batch, rank)[0]
                    preds, _ = model.module(batch, "val", batch_idx=bidx)
                    valbar.set_description(f"{bar_perfixes['val']} Epoch {epoch_idx} | " f"{model.module.evaluator}")

            recorder.record_evaluator(model.module.evaluator, epoch_idx, TrainMode.VAL)
            summarizer.summarize_evaluator(model.module.evaluator, epoch_idx, TrainMode.VAL)
            model.module.evaluator.reset_all()

    if arg.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    if arg.dist_master_port == "auto":
        arg.dist_master_port = find_free_port()
    os.environ["MASTER_PORT"] = arg.dist_master_port
    # * must have equal gpus on each node.
    arg.world_size = arg.n_gpus * arg.nodes
    # * When using a single GPU per process and per
    # * DistributedDataParallel, we need to divide the batch size
    # * ourselves based on the total number of GPUs we have
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)
    os.environ["OMP_NUM_THREADS"] = "1"

    # logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    logger.info("====> Use Distributed Data Parallel <====")
    torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)
