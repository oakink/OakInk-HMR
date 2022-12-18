from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oib.utils.builder import LOSS
from oib.utils.logger import logger

from ..utils.misc import CONST
from .criterion import TensorLoss


@LOSS.register_module
class UVD_LOSS(TensorLoss):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.FUNCTION == "mse":
            self.uvd_loss_fucntion = F.mse_loss
        elif self.cfg.FUNCTION == "smooth_l1":
            self.uvd_loss_fucntion = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss function {self.cfg.FUNCTION}")
        self.loss_lambda = self.cfg.LAMBDA

    def forward(self, preds, gt):
        gt_joints_vis = gt["target_joints_vis"]  # (B, NJ)
        gt_uvd = gt["target_joints_uvd"]  # (B, NJ, 3)
        pred_uvd = preds["uvd"][:, : CONST.NUM_JOINTS]  # (B, NJ, 3)
        pred_uvd = torch.einsum("bij,bi->bij", pred_uvd, gt_joints_vis)  # (B, NJ, 3)
        gt_uvd = torch.einsum("bij,bi->bij", gt_uvd, gt_joints_vis)  # (B, NJ, 3)

        uvd_loss = self.uvd_loss_fucntion(pred_uvd, gt_uvd)

        total_loss = self.loss_lambda * uvd_loss
        return total_loss, {"uvd_loss": uvd_loss}
