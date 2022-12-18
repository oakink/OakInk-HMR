from abc import ABC, abstractmethod

import torch.nn as nn
from oib.metrics.evaluator import Evaluator
from oib.utils.builder import build_loss, build_metric


class ModuleAbstract(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.criterions = nn.ModuleDict(self.build_criterion_list(cfg.LOSS, cfg.DATA_PRESET))
        self.evaluator: Evaluator = Evaluator(self.build_metric_list(cfg.METRIC, preset_cfg=cfg.DATA_PRESET))

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def testing_step(self, batch, batch_idx):
        pass

    def compute_loss(self, preds, gt):
        final_loss = 0.0
        final_loss_dict = {}
        for crtierion in self.criterions.values():
            loss, loss_dict = crtierion(preds, gt)
            final_loss = final_loss + loss
            final_loss_dict.update(loss_dict)
        final_loss_dict["final_loss"] = final_loss
        return final_loss, final_loss_dict

    def build_criterion_list(self, cfg, preset_cfg):
        if type(cfg) != list:
            cfg = [cfg]
        criterions = {}
        for c in cfg:
            assert c.TYPE not in criterions, f"Duplicate criterion type {c.TYPE}"
            if hasattr(self, c.TYPE):
                _c = c.clone()
                _c.defrost()
                if preset_cfg is not None:
                    _c.DATA_PRESET = preset_cfg
                _c.freeze()
                criterions[_c.TYPE] = getattr(self, _c.TYPE)(_c)
            else:
                criterions[c.TYPE] = build_loss(c, preset_cfg)
        return criterions

    def build_metric_list(self, cfg, preset_cfg):
        if type(cfg) != list:
            cfg = [cfg]
        metrics = []
        for c in cfg:
            if hasattr(self, c.TYPE):
                metrics.append(getattr(self, c.TYPE)(c))
            else:
                metrics.append(build_metric(c, preset_cfg))
        return metrics
