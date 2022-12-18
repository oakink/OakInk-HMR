import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored

from ..utils.logger import logger
from ..utils.transform import aa_to_quat
from .criterion import TensorLoss


class RleLoss3D(TensorLoss):

    def __init__(self, **cfg):
        super(RleLoss3D, self).__init__()
        self.size_average = cfg.get("SIZE_AVERAGE", True)
        self.amp = 1 / math.sqrt(2 * math.pi)
        logger.info(f"Construct {colored(type(self).__name__, 'yellow')}")

    def log_Q(self, gt_uvd, pred_uvd, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uvd - pred_uvd) / (math.sqrt(2) * sigma + 1e-9)

    def __call__(self, preds, targs, **kwargs):
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        DEVICE = final_loss.device
        pred_uvd = preds["pred_uvd"]
        pred_sigma = preds["pred_sigma"]
        log_phi = preds["log_phi"]

        gt_uvd = targs["target_joints_uvd"].reshape(pred_uvd.shape).to(DEVICE)
        Q_log_prob = self.log_Q(gt_uvd, pred_uvd, pred_sigma)

        loss = (torch.log(pred_sigma) - log_phi) + Q_log_prob

        if self.size_average:
            rle_loss = loss.sum() / len(loss)
        else:
            rle_loss = loss.mean()
        final_loss += 1.0 * rle_loss

        losses["Q_log_prob"] = Q_log_prob.mean().detach()
        losses["rle_loss"] = rle_loss.detach()
        losses[self.output_key] = final_loss
        return final_loss, losses


class ManoRleLoss(TensorLoss):

    def __init__(self, **cfg):
        super(ManoRleLoss, self).__init__()
        self.size_average = cfg.get("SIZE_AVERAGE", False)
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.lambda_joints_3d = cfg.get("LAMBDA_JOINTS_3D", 1.0)
        self.lambda_verts_3d = cfg.get("LAMBDA_VERTS_3D", 1.0)
        self.lambda_rle = cfg.get("LAMBDA_RLE", 1.0)
        self.kp_loss_type = cfg.get("KEYPOINT_LOSS_TYPE", "L1")

        if self.kp_loss_type == "L1":
            self.loss_fn = F.l1_loss
        elif self.kp_loss_type == "L2":
            self.loss_fn = F.mse_loss
        elif self.kp_loss_type == "SMOOTH_L1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        logger.info(f"Construct {colored(type(self).__name__, 'yellow')} with lambda: ")

    def log_Q(self, gt_pose, pred_pose, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_pose - pred_pose) / (math.sqrt(2) * sigma + 1e-9)

    def __call__(self, preds, targs, **kwargs):
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        DEVICE = final_loss.device

        log_phi = preds["log_phi"]
        if self.lambda_rle != 0 and log_phi is not None:
            pred_sigma = preds["pred_sigma"]
            pred_pose_aa = preds["pred_pose_aa"]
            pred_shape = preds["pred_shape"]

            gt_pose_aa = targs["target_mano_pose"].reshape(pred_pose_aa.shape).to(DEVICE)
            Q_log_prob = self.log_Q(gt_pose_aa, pred_pose_aa, pred_sigma)

            rle_loss = (torch.log(pred_sigma) - log_phi) + Q_log_prob
            if self.size_average:
                rle_loss = rle_loss.sum() / len(rle_loss)
            else:
                rle_loss = rle_loss.mean()
            final_loss += self.lambda_rle * rle_loss
        else:
            rle_loss = None

        if self.lambda_joints_3d != 0:
            gt_joints_rel = targs["target_joints_3d_rel"].to(DEVICE)
            pred_joints_rel = preds["pred_joints_rel"]
            assert pred_joints_rel.shape == gt_joints_rel.shape
            joints_3d_loss = self.loss_fn(pred_joints_rel, gt_joints_rel)
            final_loss += self.lambda_joints_3d * joints_3d_loss
        else:
            joints_3d_loss = None

        if self.lambda_verts_3d != 0:
            gt_verts_rel = targs["target_verts_3d_rel"].to(DEVICE)
            pred_verts_rel = preds["pred_verts_rel"]
            assert pred_verts_rel.shape == gt_verts_rel.shape
            verts_3d_loss = self.loss_fn(pred_verts_rel, gt_verts_rel)
            final_loss += self.lambda_verts_3d * verts_3d_loss
        else:
            verts_3d_loss = None

        losses["rle_loss"] = rle_loss
        losses["j3d_loss"] = joints_3d_loss
        losses["v3d_loss"] = verts_3d_loss

        losses[self.output_key] = final_loss
        return final_loss, losses
