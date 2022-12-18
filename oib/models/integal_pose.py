import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oib.criterions.criterion import TensorLoss
from oib.metrics.basic_metric import Metric, VisMetric
from oib.metrics.evaluator import Evaluator
from oib.models.model_abstraction import ModuleAbstract
from oib.viztools.draw import concat_imgs, draw_batch_joint_images

from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import CONST, enable_lower_param, param_size
from ..utils.transform import batch_uvd2xyz
from .layers import create_backbone


def norm_heatmap(norm_type: str, heatmap: torch.Tensor) -> torch.Tensor:
    """
    Args:
        norm_type: str: either in [softmax, sigmoid, divide_sum],
        heatmap: TENSOR (BATCH, C, ...)

    Returns:
        TENSOR (BATCH, C, ...)
    """
    shape = heatmap.shape
    if norm_type == "softmax":
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == "sigmoid":
        return heatmap.sigmoid()
    else:
        raise NotImplementedError


def integral_heatmap3d(heatmap3d: torch.Tensor) -> torch.Tensor:
    """
    Integral 3D heatmap into whd corrdinates. u stand for the prediction in WIDTH dimension
    ref: https://arxiv.org/abs/1711.08229

    Args:
        heatmap3d: TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH) d,v,u

    Returns:
        uvd: TENSOR (BATCH, NCLASSES, 3) RANGE:0~1
    """
    d_accu = torch.sum(heatmap3d, dim=[3, 4])
    v_accu = torch.sum(heatmap3d, dim=[2, 4])
    u_accu = torch.sum(heatmap3d, dim=[2, 3])

    weightd = torch.arange(d_accu.shape[-1], dtype=d_accu.dtype, device=d_accu.device) / d_accu.shape[-1]
    weightv = torch.arange(v_accu.shape[-1], dtype=v_accu.dtype, device=v_accu.device) / v_accu.shape[-1]
    weightu = torch.arange(u_accu.shape[-1], dtype=u_accu.dtype, device=u_accu.device) / u_accu.shape[-1]

    d_ = d_accu.mul(weightd)
    v_ = v_accu.mul(weightv)
    u_ = u_accu.mul(weightu)

    d_ = torch.sum(d_, dim=-1, keepdim=True)
    v_ = torch.sum(v_, dim=-1, keepdim=True)
    u_ = torch.sum(u_, dim=-1, keepdim=True)

    uvd = torch.cat([u_, v_, d_], dim=-1)
    return uvd  # TENSOR (BATCH, NCLASSES, 3)


class IntegralDeconvHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.inplanes = cfg.INPUT_CHANNEL
        self.depth_res = cfg.HEATMAP_3D_SIZE[2]
        self.height_res = cfg.HEATMAP_3D_SIZE[1]
        self.width_res = cfg.HEATMAP_3D_SIZE[0]
        self.deconv_with_bias = cfg.DECONV_WITH_BIAS
        self.nclasses = cfg.N_CLASSES
        self.norm_type = cfg.NORM_TYPE

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.NUM_DECONV_LAYERS,
            cfg.NUM_DECONV_FILTERS,
            cfg.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.N_CLASSES * self.depth_res,
            kernel_size=cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0,
        )
        self.init_weights()

    def init_weights(self):
        logger.info("=> init deconv weights from normal distribution")
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info("=> init final conv weights from normal distribution")
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def view_to_bcdhw(self, x: torch.Tensor) -> torch.Tensor:
        """
        view a falttened 2D heatmap to 3D heatmap, sharing the same memory by using view()
        Args:
            x: TENSOR (BATCH, NCLASSES * DEPTH, HEIGHT|ROWS, WIDTH|COLS)

        Returns:
            TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        """
        return x.contiguous().view(
            x.shape[0],  # BATCH,
            self.nclasses,  # NCLASSES
            self.depth_res,  # DEPTH
            self.height_res,  # HEIGHT,
            self.width_res,  # WIDTH
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError()

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(num_kernels), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs["feature"]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        x = x.reshape((x.shape[0], self.nclasses, -1))  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)
        x = norm_heatmap(self.norm_type, x)  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)

        confi = torch.max(x, dim=-1).values  # TENSOR (B, NCLASS)
        assert x.dim() == 3, f"Unexpected dim, expect x has shape (B, C, DxHxW), got {x.shape}"
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-7)
        x = self.view_to_bcdhw(x)  # TENSOR(BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        x = integral_heatmap3d(x)  # TENSOR (BATCH, NCLASSES, 3)
        return {"uvd": x, "uvd_confi": confi}


@MODEL.register_module
class IntegralPose(ModuleAbstract):
    @enable_lower_param
    def __init__(self, cfg):
        super(IntegralPose, self).__init__(cfg)
        self.name = type(self).__name__
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.inp_res = cfg.DATA_PRESET.IMAGE_SIZE
        self.train_mode = cfg.MODE
        assert self.train_mode in ["3D", "UVD_ortho", "UVD"], f"Model's mode mismatch, got {self.train_mode}"

        # ***** build network *****
        self.backbone = create_backbone(cfg.BACKBONE)
        self.pose_head = IntegralDeconvHead(cfg.HEAD)

        if cfg.BACKBONE.PRETRAINED and cfg.PRETRAINED:
            logger.warning(f"{self.name}'s backbone {cfg.BACKBONE.TYPE} re-initalized by {cfg.PRETRAINED}")
        self.init_weights(pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    # ***** forward *****

    def forward(self, inputs, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _forward_impl(self, inputs):
        x = inputs["image"]
        feat = self.backbone(image=x)
        res = self.pose_head(feature=feat["res_layer4"])
        return res

    # ***** each step *****

    def training_step(self, batch, **kwargs):
        # forward your network
        preds = self._forward_impl(batch)

        # compute loss
        final_loss, final_loss_dict = self.compute_loss(preds, batch)

        batch_size = batch["image"].shape[0]

        joints_uvd = preds["uvd"]
        inp_res = torch.Tensor(self.inp_res).to(joints_uvd.device)
        joints_2d = torch.einsum("bij,j->bij", joints_uvd[:, :, :2], inp_res)
        preds["joints_2d"] = joints_2d

        if self.cfg.MODE in ["3D", "UVD_ortho"]:
            cam_mode = "ortho" if self.cfg.MODE == "UVD_ortho" else "persp"
            intr = batch["target_ortho_intr"] if cam_mode == "ortho" else batch["target_cam_intr"]
            joints_3d = batch_uvd2xyz(
                uvd=joints_uvd,
                root_joint=batch["target_root_joint"],
                intr=intr,
                inp_res=self.inp_res,
                depth_range=CONST.UVD_DEPTH_RANGE,
                camera_mode=cam_mode,
            )
            preds["joints_3d"] = joints_3d

        elif self.cfg.MODE == "UVD":
            # * YT3D only has gt annotaiton of UV & D, no xyz.
            # * When caculating mpepe, we use "target_joints_uvd" and "uvd" as the key to compute the score.
            pass

        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)

        return preds, final_loss_dict

    def validation_step(self, batch, **kwargs):
        preds = self._forward_impl(batch)
        final_loss_dict = {}  # * we don't need to compute loss for validation
        return preds, final_loss_dict

    def testing_step(self, batch, **kwargs):
        preds = self._forward_impl(batch)
        batch_size = batch["image"].shape[0]
        joints_uvd = preds["uvd"]
        inp_res = torch.Tensor(self.inp_res).to(joints_uvd.device)
        joints_2d = torch.einsum("bij,j->bij", joints_uvd[:, :, :2], inp_res)
        preds["joints_2d"] = joints_2d

        if self.cfg.MODE in ["3D", "UVD_ortho"]:
            camera_mode = "ortho" if self.cfg.MODE == "UVD_ortho" else "persp"
            intr = batch["target_ortho_intr"] if camera_mode == "ortho" else batch["target_cam_intr"]
            joints_3d = batch_uvd2xyz(
                uvd=joints_uvd,
                root_joint=batch["target_root_joint"],
                intr=intr,
                inp_res=self.inp_res,
                depth_range=CONST.UVD_DEPTH_RANGE,
                camera_mode=camera_mode,
            )
            preds["joints_3d"] = joints_3d
        elif self.cfg.MODE == "UVD":
            # some dataset (eg. YT3D) only has gt annotaiton of UV&D, no xyz.
            pass

        if kwargs.get("disable_evaluator", False):
            final_loss_dict = {}
        else:
            final_loss_dict = {}  # * we don't need to compute loss for testing
            with torch.no_grad():
                self.evaluator.feed_all(preds, batch, final_loss_dict)

        if "callback" in kwargs:
            kwargs["callback"](preds=preds, inputs=batch)

        return preds, final_loss_dict

    # ***** network loss *****
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
            pred_uvd = preds["uvd"]  # (B, NJ, 3)
            pred_uvd = torch.einsum("bij,bi->bij", pred_uvd, gt_joints_vis)  # (B, NJ, 3)
            gt_uvd = torch.einsum("bij,bi->bij", gt_uvd, gt_joints_vis)  # (B, NJ, 3)

            uvd_loss = self.uvd_loss_fucntion(pred_uvd, gt_uvd)

            total_loss = self.loss_lambda * uvd_loss
            return total_loss, {"uvd_loss": uvd_loss}

    # ***** network metrics *****
    class Integal_Pose_Vis_Metric(VisMetric):
        def feed(self, preds, targs):
            if self.images is not None:  # * only visualize the first batch
                return
            img = draw_batch_joint_images(preds["joints_2d"], targs["target_joints_2d"], targs["image"])
            self.images = [concat_imgs(img)]

    # ***** network initializer *****

    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {self.name} weights in backbone and head")
            """
            Add init for other modules
            ...
            """
        elif os.path.isfile(pretrained):
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
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"=> Loading SUCCEEDED")
        else:
            logger.error(f"=> No {self.name} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
