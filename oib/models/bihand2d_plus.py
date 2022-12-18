# Copyright (c) Lixin YANG. All Rights Reserved.
"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""
import os
from collections import OrderedDict

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oib.metrics.basic_metric import AverageMeter, LossMetric
from oib.models.model_abstraction import ModuleAbstract
from oib.utils.builder import MODEL
from oib.utils.heatmap import accuracy_heatmap, get_heatmap_pred
from oib.utils.logger import logger
from oib.utils.misc import param_size
from oib.utils.transform import bchw_2_bhwc, denormalize
from oib.viztools.draw import plot_image_heatmap_mask, plot_image_joints_mask

from .layers.hourglass import BottleneckX2, HourglassBisected


class bihand_2d_loss(nn.Module):
    def __init__(self, cfg):
        super(bihand_2d_loss, self).__init__()
        self.lambda_heatmap = 100.0
        self.lambda_mask = 1.0

    def forward(self, preds, targs, **kwargs):
        batch_size = targs["target_joints_vis"].shape[0]
        kpvis = targs["target_joints_vis"][..., None]  # (B, 21, 1)
        DEVICE = kpvis.device
        losses = {}
        gt_heatmap = targs["target_joints_heatmap"]

        final_loss = torch.Tensor([0]).to(DEVICE)
        heatmap_loss = torch.Tensor([0]).to(DEVICE)

        for heatmap in preds["heatmaps_list"]:
            njoints = heatmap.size(1)
            heatmap = heatmap.reshape((batch_size, njoints, -1))  # (B, 21, H*W)
            gt_heatmap_ = gt_heatmap.reshape((batch_size, njoints, -1))  # (B, 21, H*W)
            hmloss = 0.5 * F.mse_loss(heatmap, gt_heatmap_, reduction="none")  # (B, 21, H*W)
            heatmap_loss += (kpvis * hmloss).mean()

        final_loss += self.lambda_heatmap * heatmap_loss
        losses["heatmap_loss"] = heatmap_loss

        mask_loss = torch.Tensor([0]).to(DEVICE)
        gt_mask = targs["mask"].to(DEVICE)
        for mask in preds["masks_list"]:
            mask = mask.view(batch_size, -1)  # (B, 64x64)
            gt_mask_ = gt_mask.view(batch_size, -1)

            mloss = F.binary_cross_entropy_with_logits(mask, gt_mask_, reduction="none")
            mloss = torch.sum(mloss, dim=1) / mask.shape[1]
            mloss = torch.sum(mloss)
            mloss = torch.sum(mloss) / batch_size
            mask_loss += mloss

        final_loss += self.lambda_mask * mask_loss
        losses["mask_loss"] = mask_loss

        losses["loss"] = final_loss
        return final_loss, losses


@MODEL.register_module
class BiHand2DPlus(ModuleAbstract):
    def __init__(self, cfg):
        super(BiHand2DPlus, self).__init__()
        self.name = "BiHand2DPlus"
        self.cfg = cfg
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL

        self.bihand_2d_loss = bihand_2d_loss(cfg)
        self.loss_metric = LossMetric(cfg)

        self.heatmap_acc = AverageMeter(name="heatmap_acc")

        # ***** build network *****
        block = BottleneckX2
        self.njoints = cfg.N_JOINTS
        self.nstacks = cfg.N_STACKS
        nblocks = cfg.N_BLOCKS
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer1 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 64 * 2 = 128
        self.layer2 = self._make_residual(block, nblocks, self.in_planes, 2 * self.in_planes)
        # current self.in_planes is 128 * 2 = 256
        self.layer3 = self._make_residual(block, nblocks, self.in_planes, self.in_planes)

        ch = self.in_planes  # 256

        hg2b, res1, res2, fc1, _fc1, fc2, _fc2 = [], [], [], [], [], [], []
        hm, _hm, mask, _mask = [], [], [], []
        for i in range(self.nstacks):  # 2
            hg2b.append(HourglassBisected(block, nblocks, ch, depth=4))
            res1.append(self._make_residual(block, nblocks, ch, ch))
            res2.append(self._make_residual(block, nblocks, ch, ch))
            fc1.append(self._make_fc(ch, ch))
            fc2.append(self._make_fc(ch, ch))
            hm.append(nn.Conv2d(ch, self.njoints, kernel_size=1, bias=True))
            mask.append(nn.Conv2d(ch, 1, kernel_size=1, bias=True))

            if i < self.nstacks - 1:
                _fc1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _fc2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=False))
                _hm.append(nn.Conv2d(self.njoints, ch, kernel_size=1, bias=False))
                _mask.append(nn.Conv2d(1, ch, kernel_size=1, bias=False))

        self.hg2b = nn.ModuleList(hg2b)  # hgs: hourglass stack
        self.res1 = nn.ModuleList(res1)
        self.fc1 = nn.ModuleList(fc1)
        self._fc1 = nn.ModuleList(_fc1)
        self.res2 = nn.ModuleList(res2)
        self.fc2 = nn.ModuleList(fc2)
        self._fc2 = nn.ModuleList(_fc2)
        self.hm = nn.ModuleList(hm)
        self._hm = nn.ModuleList(_hm)
        self.mask = nn.ModuleList(mask)
        self._mask = nn.ModuleList(_mask)

        if cfg.PRETRAINED:
            self.init_weights(pretrained=cfg.PRETRAINED)
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def init_weights(self, pretrained=None):
        if os.path.isfile(pretrained):
            logger.info(f"=> Loading {self.name} pretrained model from: {pretrained}")
            ### self.load_state_dict(pretrained_state_dict, strict=False)
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
        else:
            logger.error(f"=> No {self.name} checkpoints file found in {pretrained}")
            raise FileNotFoundError()

    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(in_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        return nn.Sequential(conv, bn, self.relu)

    def _make_residual(self, block, nblocks, in_planes, out_planes):
        layers = []
        layers.append(block(in_planes, out_planes))
        self.in_planes = out_planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def compute_loss(self, preds, batch):
        return self.bihand_2d_loss(preds, batch)

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def _forward_impl(self, inputs):
        x = inputs["image"]
        l_hm, l_mask = [], []
        x = self.conv1(x)  # x: (N,64,128,128)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)  # x: (N,128,64,64)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.nstacks):  # 2
            y_1, y_2, _ = self.hg2b[i](x)

            y_1 = self.res1[i](y_1)
            y_1 = self.fc1[i](y_1)
            est_hm = self.hm[i](y_1)
            l_hm.append(est_hm)

            y_2 = self.res2[i](y_2)
            y_2 = self.fc2[i](y_2)
            est_mask = self.mask[i](y_2)
            l_mask.append(est_mask)

            if i < self.nstacks - 1:
                _fc1 = self._fc1[i](y_1)
                _hm = self._hm[i](est_hm)
                _fc2 = self._fc2[i](y_2)
                _mask = self._mask[i](est_mask)
                x = x + _fc1 + _fc2 + _hm + _mask
        assert len(l_hm) == self.nstacks

        res = {"heatmaps_list": l_hm, "masks_list": l_mask}
        return res

    def training_step(self, batch, step_idx):
        batch_size = batch["image"].shape[0]
        preds = self._forward_impl(batch)
        loss, loss_dict = self.compute_loss(preds, batch)
        self.loss_metric.feed(loss_dict, batch_size)

        if step_idx % self.train_log_interval == 0 and self.summary is not None:
            self.summary.add_scalar("loss", loss.item(), step_idx)
            self.summary.add_scalar("heatmap_loss", loss_dict["heatmap_loss"].item(), step_idx)
            self.summary.add_scalar("mask_loss", loss_dict["mask_loss"].item(), step_idx)

            viz_interval = self.train_log_interval * 20
            if step_idx % viz_interval == 0:  # viz a image
                rand_i = np.random.randint(0, batch_size)
                heatmap = preds["heatmaps_list"][-1][rand_i].unsqueeze(0)  # (1, 21, H, W)
                mask = preds["masks_list"][-1][rand_i].unsqueeze(0)  # (1, 1, H, W)
                image = batch["image"][rand_i].unsqueeze(0)  # (1, 3, H, W)

                image = bchw_2_bhwc(denormalize(image.detach().cpu(), [0.5, 0.5, 0.5], [1, 1, 1], inplace=False))
                image = image.mul_(255.0).numpy().astype(np.uint8)  # (B, H, W, 3)
                mask = mask.detach().cpu().mul_(255.0).clamp(0, 255).squeeze(1).numpy().astype(np.uint8)  # (B, H, W)
                heatmap = heatmap.detach().cpu().numpy().astype(np.float32)  # (B, 21, H, W)
                draw = plot_image_heatmap_mask(image[0], heatmap[0], mask[0])
                self.summary.add_image("train_image_heatmap_mask", draw, step_idx, dataformats="HWC")

        return preds, loss_dict

    def on_train_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-train-"  # BiHand2D-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validation_step(self, batch, step_idx):
        BATCH_SIZE = batch["image"].shape[0]
        preds = self._forward_impl(batch)

        # do some evaluation & accuracy
        heatmap = preds["heatmaps_list"][-1]  # (B, 21, H, W)
        mask = preds["masks_list"][-1]  # (B, 1, H, W)
        image = batch["image"]  # (B, 3, H, W)
        gt_heatmap = batch["target_joints_heatmap"]
        kp_vis = batch["target_joints_vis"]  # (B, 21)

        acc, per_jt_acc = accuracy_heatmap(heatmap, gt_heatmap, kp_vis)
        self.heatmap_acc.update_by_mean(acc, BATCH_SIZE)

        pred_jts, confi = get_heatmap_pred(heatmap)  # (B, njoint, 2)
        heatmap_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        img_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        rescale = torch.Tensor([img_size[0] / heatmap_size[0], img_size[1] / heatmap_size[1]]).to(heatmap.device)
        pred_jts = torch.einsum("bij,j->bij", pred_jts, rescale)

        preds["joints_2d"] = pred_jts
        preds["joints_2d_confi"] = confi

        if step_idx % self.train_log_interval == 0 and self.summary is not None:
            self.summary.add_scalar("heatmap_acc", self.heatmap_acc.avg, step_idx)

            image = bchw_2_bhwc(denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=True))
            image = image.mul_(255.0).cpu().numpy().astype(np.uint8)  # (B, H, W, 3)
            mask = mask.mul_(255.0).clamp(0, 255).squeeze().detach().cpu().numpy().astype(np.uint8)  # (B, H, W)
            pred_jts = pred_jts.detach().cpu().numpy().astype(np.float32)  # (B, njoint, 2)
            img_list = []
            for im, j2d, m in zip(image, pred_jts, mask):
                comb = plot_image_joints_mask(im, j2d, m)
                img_list.append(comb)

            draw = torch.from_numpy(np.stack(img_list, axis=0))
            self.summary.add_images("val_image_joints_mask", draw, step_idx, dataformats="NHWC")

        return preds, {"heatmap_accuracy": acc}

    def on_val_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-val"
        recorder.record_metric([self.heatmap_acc], epoch_idx, comment=comment)

        self.heatmap_acc.reset()

    def forward(self, inputs, step_idx, mode="train"):
        if mode == "train":
            return self.training_step(inputs, step_idx)
        elif mode == "val":
            return self.validation_step(inputs, step_idx)
        elif mode == "test":
            return self.testing_setp(inputs, step_idx)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx)
        else:
            raise ValueError(f"Unknown mode {mode}")
