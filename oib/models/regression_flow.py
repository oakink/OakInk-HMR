import math
import os

import numpy as np
import torch
import torch.nn as nn
from oib.metrics.basic_metric import VisMetric
from oib.utils.builder import MODEL
from oib.utils.logger import logger
from oib.utils.misc import CONST, enable_lower_param, param_size
from oib.utils.transform import batch_uvd2xyz, rot6d_to_aa
from oib.viztools.draw import draw_batch_joint_images, concat_imgs

from .layers import create_backbone
from .layers.mano_wrapper import MANO
from .layers.real_nvp import RealNVP
from .model_abstraction import ModuleAbstract


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / (x_norm + 1e-7)

        if self.bias:
            y = y + self.linear.bias
        return y


@MODEL.register_module
class RegressFlow3D(ModuleAbstract):
    """Res-Loglikelihood Estimation https://"""

    def __init__(self, cfg) -> None:
        super(RegressFlow3D, self).__init__(cfg)
        self.name = "RegressFlow3D"
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.mode = cfg.MODE
        assert self.mode in ["3D", "UVD_ortho"], f"Mode must be [3D, UVD_ortho], got {self.mode}"

        self.fc_dim = cfg.NUM_FC_FILTERS
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.inp_res = cfg.DATA_PRESET.IMAGE_SIZE
        self.preact = create_backbone(cfg.BACKBONE)
        self.num_joints = CONST.NUM_JOINTS

        self.feature_channel = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}[
            cfg.BACKBONE.TYPE
        ]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcs, out_channel = self._make_fc_layer()
        self.fc_coord = Linear(out_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(out_channel, self.num_joints * 3)
        self.fc_layers = [self.fc_coord, self.fc_sigma]
        self.flow3d = RealNVP(prior_dim=3)

        self.init_weights(pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def _make_fc_layer(self):
        fc_layers = []
        nfc = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(nfc):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())

        return nn.Sequential(*fc_layers), input_channel

    def training_step(self, batch, **kwargs):
        """_summary_

        Args:
            batch (_type_): _description_
            step_idx (_type_): _description_
        """
        img = batch["image"]
        # BATCH_SIZE = img.shape[0]
        preds = self._forward_impl(batch, is_train=True)
        final_loss, final_loss_dict = self.compute_loss(preds, gt=batch)

        # convert from uvd to joints xyz, needs to know the root_d and intr.
        uvd = preds["pred_uvd"]
        inp_res = [img.shape[3], img.shape[2]]  # [H, W]
        gt_root_jts = batch["target_root_joint"]  # (B, 3)
        if self.mode == "UVD_ortho":
            gt_ortho_intr = batch["target_ortho_intr"]
            pred_jts = batch_uvd2xyz(
                uvd,
                gt_root_jts,
                gt_ortho_intr,
                inp_res,
                CONST.UVD_DEPTH_RANGE,
                camera_mode="ortho",
            )
        else:
            gt_intr = batch["target_cam_intr"]
            pred_jts = batch_uvd2xyz(uvd, gt_root_jts, gt_intr, inp_res, CONST.UVD_DEPTH_RANGE)
        pred_jts_rel = pred_jts - gt_root_jts.unsqueeze(1)
        preds["joints_3d"] = pred_jts
        preds["joints_3d_rel"] = pred_jts_rel

        inp_res = torch.Tensor(self.inp_res).to(uvd.device)
        pred_2d = torch.einsum("bij,j->bij", uvd[:, :, :2], inp_res)
        preds["joints_2d"] = pred_2d

        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)

        return preds, final_loss_dict

    def validation_step(self, batch, **kwargs):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        preds = self._forward_impl(batch, is_train=False)
        return preds, {}

    def testing_step(self, batch, **kwargs):
        preds = self._forward_impl(batch, is_train=False)

        img = batch["image"]
        # BATCH_SIZE = img.shape[0]
        uvd = preds["pred_uvd"]
        gt_root_jts = batch["target_root_joint"]  # (B, 3)
        # gt_intr = batch["target_cam_intr"]
        inp_res = [img.shape[3], img.shape[2]]  # [H, W]
        if self.mode == "UVD_ortho":
            gt_ortho_intr = batch["target_ortho_intr"]
            pred_jts = batch_uvd2xyz(
                uvd,
                gt_root_jts,
                gt_ortho_intr,
                inp_res,
                camera_mode="ortho",
            )
        else:
            gt_intr = batch["target_cam_intr"]
            pred_jts = batch_uvd2xyz(uvd, gt_root_jts, gt_intr, inp_res, CONST.UVD_DEPTH_RANGE)
        pred_jts_rel = pred_jts - gt_root_jts.unsqueeze(1)
        preds["joints_3d"] = pred_jts
        preds["joints_3d_rel"] = pred_jts_rel

        inp_res = torch.Tensor(self.inp_res).to(uvd.device)
        pred_2d = torch.einsum("bij,j->bij", uvd[:, :, :2], inp_res)
        preds["joints_2d"] = pred_2d

        if kwargs.get("disable_evaluator", False):
            final_loss_dict = {}
        else:
            final_loss_dict = {}  # * we don't need to compute loss for testing
            with torch.no_grad():
                self.evaluator.feed_all(preds, batch, final_loss_dict)

        if "callback" in kwargs:
            kwargs["callback"](preds=preds, inputs=batch)

        return preds, {}

    def forward(self, inputs, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _forward_impl(self, inputs, is_train=True):
        x = inputs["image"]  # (B, C, H, W)
        labels = inputs
        BATCH_SIZE = x.shape[0]

        _feat = self.preact(image=x)["res_layer4"]
        feat = self.avg_pool(_feat).reshape(BATCH_SIZE, -1)

        pred_uvd = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 3)
        if not is_train:  # remove the center D shift
            pred_uvd[:, :, 2] = pred_uvd[:, :, 2] - pred_uvd[:, self.center_idx, 2].unsqueeze(1) + 0.5

        pred_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, 3)
        pred_sigma = torch.sigmoid(pred_sigma) + 1e-7
        scores = 1 - pred_sigma
        scores = torch.mean(scores, dim=2, keepdim=True)

        if is_train:
            gt_uvd = labels["target_joints_uvd"].reshape(pred_uvd.shape)
            gt_uvd_mask = labels["target_joints_vis"].unsqueeze(-1).repeat(1, 1, 3)

            assert pred_uvd.shape == pred_sigma.shape, (pred_uvd.shape, pred_sigma.shape)

            bar_mu = (pred_uvd - gt_uvd) / pred_sigma
            bar_mu = bar_mu.reshape(-1, 3)  # (BxN, 3)

            bar_mu_3d = bar_mu
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi = log_phi_3d.reshape(BATCH_SIZE, self.num_joints, 1)
        else:
            log_phi = None

        res = {
            "pred_uvd": pred_uvd,
            "pred_sigma": pred_sigma,
            "log_phi": log_phi,
            "maxvals": scores.float(),
        }
        return res

    # ***** network loss *****
    class RLE_Loss(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.size_average = cfg.SIZE_AVERAGE
            self.amp = 1 / math.sqrt(2 * math.pi)
            self.lambda_rle = 1.0

        def log_Q(self, gt_uvd, pred_uvd, sigma):
            return torch.log(sigma / self.amp) + torch.abs(gt_uvd - pred_uvd) / (math.sqrt(2) * sigma + 1e-7)

        def forward(self, preds, gt, **kwargs):
            pred_uvd = preds["pred_uvd"]
            pred_sigma = preds["pred_sigma"]
            log_phi = preds["log_phi"]

            gt_uvd = gt["target_joints_uvd"].reshape(pred_uvd.shape)
            Q_log_prob = self.log_Q(gt_uvd, pred_uvd, pred_sigma)

            loss = (torch.log(pred_sigma) - log_phi) + Q_log_prob

            if self.size_average:
                rle_loss = loss.sum() / len(loss)
            else:
                rle_loss = loss.mean()

            loss_dict = {}
            final_loss = self.lambda_rle * rle_loss

            loss_dict["Q_log_prob"] = Q_log_prob.mean().detach()
            loss_dict["rle_loss"] = rle_loss.detach()
            loss_dict["loss"] = final_loss
            return final_loss, loss_dict

    # ***** network metrics *****
    class RLE_Vis_Metric(VisMetric):
        def feed(self, preds, targs):
            if self.images is not None:  # * only visualize the first batch
                return
            img = draw_batch_joint_images(preds["joints_2d"], targs["target_joints_2d"], targs["image"])
            self.images = [concat_imgs(img)]

    def init_weights(self, pretrained=None):
        if pretrained is None or pretrained == "":
            for m in self.fcs:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
            for m in self.fc_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
            self.flow3d._init()
        elif os.path.isfile(pretrained):
            from collections import OrderedDict

            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {self.name} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"=> Loading SUCCEEDED")
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
