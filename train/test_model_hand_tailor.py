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
from oib.opt import parse_exp_args
from oib.utils import builder
from oib.utils.config import get_config
from oib.utils.etqdm import etqdm
from oib.utils.logger import logger
from oib.utils.misc import CONST, TrainMode, bar_perfixes, format_args_cfg
from oib.utils.netutils import clip_gradient, setup_seed
from oib.utils.recorder import Recorder
from oib.utils.summarizer import Summarizer
from oib.metrics.evaluator import Evaluator


class HandTailorContext:
    def __init__(self, model_type, preset_cfg):
        self.model_type = model_type

        if self.model_type in ["IntegralPose", "RegressFlow3D"]:
            # setup jax
            import jax

            jax.config.update("jax_platform_name", "cpu")

            # setup fitting unit
            from oib.postprocess.hand_tailor.iknet import FittingUnit

            self.fitting_unit = FittingUnit()
        else:
            self.fitting_unit = None

        # setup evaluator
        from oib.utils.config import CN_R
        from oib.utils.builder import build_metric

        metric_cfg = [
            CN_R(
                {
                    "TYPE": "LossMetric",
                    "VIS_LOSS_KEYS": [],
                }
            ),
            CN_R({"TYPE": "MeanEPE", "NAME": "joints_3d", "PRED_KEY": "joints_3d", "GT_KEY": "target_joints_3d"}),
            CN_R({"TYPE": "MeanEPE2", "NAME": "verts_3d", "PRED_KEY": "verts_3d", "GT_KEY": "target_verts_3d"}),
            CN_R({"TYPE": "Hand3DPCK", "EVAL_TYPE": "joints_3d", "VAL_MIN": 0.00, "VAL_MAX": 0.05, "STEPS": 20}),
            CN_R({"TYPE": "PAEval", "MESH_SCORE": True}),
        ]
        extra_metric_cls = {}
        if self.model_type in ["IntegralPose"]:
            from oib.models.integal_pose import IntegralPose

            metric_cfg.append(CN_R({"TYPE": "Integal_Pose_Vis_Metric"}))
            extra_metric_cls["Integal_Pose_Vis_Metric"] = IntegralPose.Integal_Pose_Vis_Metric
        elif self.model_type in ["RegressFlow3D"]:
            from oib.models.regression_flow import RegressFlow3D

            metric_cfg.append(CN_R({"TYPE": "RLE_Vis_Metric"}))
            extra_metric_cls["RLE_Vis_Metric"] = RegressFlow3D.RLE_Vis_Metric
        elif self.model_type in ["I2L_MeshNet"]:
            from oib.external.i2l_meshnet.model import I2L_MeshNet
            
            metric_cfg.append(CN_R({"TYPE": "I2L_MeshNet_Vis_Metric"}))
            extra_metric_cls["I2L_MeshNet_Vis_Metric"] = I2L_MeshNet.I2L_MeshNet_Vis_Metric

        metric_list = []
        for c in metric_cfg:
            if c.TYPE in extra_metric_cls:
                _metric = extra_metric_cls[c.TYPE](c)
            else:
                _metric = build_metric(c, preset_cfg=preset_cfg)
            metric_list.append(_metric)
        self.evaluator: Evaluator = Evaluator(metric_list)

    def __call__(self, preds, inputs):
        # adapt inp to fitting unit
        inp = {
            "image": inputs["image"],
            "cam_intr": inputs["cam_intr"],
        }
        pred_joints = preds["joints_3d"]
        if self.fitting_unit is not None:
            v_list, j_list = self.fitting_unit(inp, pred_joints)
            preds["verts_3d"] = torch.from_numpy(np.array(v_list)).to(inputs["verts_3d"])  # add verts_3d field
        with torch.no_grad():
            self.evaluator.feed_all(preds, inputs, {})


def _init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
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
        worker_init_fn=_init_fn,
    )

    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET, train_cfg=cfg.TRAIN)
    model = DP(model).to(arg.device)

    # hand_tailor
    hand_tailor_ctx = HandTailorContext(model_type=cfg.MODEL.TYPE, preset_cfg=cfg.DATA_PRESET)

    with torch.no_grad():
        model.eval()
        testbar = etqdm(test_loader, rank=rank)
        for bidx, batch in enumerate(testbar):
            preds, _ = model(batch, "test", callback=hand_tailor_ctx, disable_evaluator=True)
            testbar.set_description(f"{bar_perfixes['test']} Epoch {0} | " f"{model.module.evaluator}")

        recorder.record_evaluator(hand_tailor_ctx.evaluator, 0, TrainMode.TEST)
        summarizer.summarize_evaluator(hand_tailor_ctx.evaluator, 0, TrainMode.TEST)


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    assert (
        arg.n_gpus == 1
    ), """
        We only support a single GPU when testing, or the SCORES WILL BE WRONG!!!
        If the GPU is out of memory, you can try to reduce the batch size.
    """
    assert arg.reload is not None, "reload checkpointint path is required"
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ["OMP_NUM_THREADS"] = "1"

    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    # logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")

    logger.info("====> Testing on single GPU (Data Parallel) <====")
    main_worker(cfg, arg, exp_time)
