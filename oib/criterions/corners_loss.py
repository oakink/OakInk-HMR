import torch
import torch.nn.functional as F

from oib.utils.builder import LOSS

from .criterion import TensorLoss


@LOSS.register_module
class CORNERS_LOSS(TensorLoss):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.FUNCTION == "mse":
            self.loss_fucntion = F.mse_loss
        elif self.cfg.FUNCTION == "smooth_l1":
            self.loss_fucntion = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss function {self.cfg.FUNCTION}")
        self.loss_lambda = self.cfg.LAMBDA

    def forward(self, preds, gt):
        gt_corners_vis = gt["target_corners_vis"]  # (B, 8)
        gt_corners = gt["target_corners_3d"]  # (B, 8, 3)
        pred_corners = preds["pred_corners_3d"]  # (B, 8, 3)
        pred_c = torch.einsum("bij,bi->bij", pred_corners, gt_corners_vis)  # (B, 8, 3)
        gt_c = torch.einsum("bij,bi->bij", gt_corners, gt_corners_vis)  # (B, 8, 3)

        uvd_loss = self.loss_fucntion(pred_c, gt_c)

        total_loss = self.loss_lambda * uvd_loss
        return total_loss, {"corners_loss": uvd_loss}
