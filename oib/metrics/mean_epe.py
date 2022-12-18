from abc import ABC, abstractmethod
from typing import Dict

import torch
from oib.utils.builder import METRIC

from ..utils.logger import logger
from ..utils.misc import CONST
from .basic_metric import AverageMeter, Metric


@METRIC.register_module
class MeanEPE(Metric):
    def __init__(self, cfg) -> None:
        super(MeanEPE, self).__init__(cfg)
        self._num_kp = CONST.NUM_JOINTS
        self.avg_meter = AverageMeter()
        self.name = f"{cfg.NAME}_mepe"
        self.pred_key = cfg.PRED_KEY
        self.gt_key = cfg.GT_KEY
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, preds, targs, kp_vis=None, **kwargs):
        pred_kp = preds[self.pred_key]
        gt_kp = targs[self.gt_key]
        assert len(pred_kp.shape) == 3, logger.error(
            "X pred shape, should as (BATCH, NPOINTS, 1|2|3)"
        )  # TENSOR (BATCH, NPOINTS, 2|3)

        diff = pred_kp - gt_kp  # TENSOR (B, N, 1|2|3)
        dist_ = torch.norm(diff, p="fro", dim=2)  # TENSOR (B, N)
        dist_batch = torch.mean(dist_, dim=1, keepdim=True)  # TENSOR (B, 1)
        BATCH_SIZE = dist_batch.shape[0]
        sum_dist_batch = torch.sum(dist_batch)
        self.avg_meter.update(sum_dist_batch.item(), n=BATCH_SIZE)

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        avg = self.avg_meter.avg
        measures[f"{self.name}"] = avg
        return measures

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"


@METRIC.register_module
class MeanEPE2(MeanEPE):
    pass
