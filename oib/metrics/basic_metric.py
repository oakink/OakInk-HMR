from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from oib.utils.builder import METRIC


class Metric(ABC):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.skip = False

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def feed(self, preds, targs, **kwargs):
        pass

    @abstractmethod
    def get_measures(self, **kwargs) -> Dict:
        pass


class VisMetric(Metric):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.images = None

    def reset(self):
        del self.images
        self.images = None

    def get_measures(self, **kwargs) -> Dict:
        return {"image": self.images}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="") -> None:
        super().__init__()
        self.reset()
        self.name = name

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_by_mean(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.avg:.4e}"

    def get_measures(self) -> Dict:
        return {self.name + "avg": self.avg}


@METRIC.register_module
class LossMetric(Metric):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.count = 0
        self._losses: Dict[str, AverageMeter] = {}
        self._vis_loss_keys: List[str] = cfg["VIS_LOSS_KEYS"]

    def reset(self):
        self._losses: Dict[str, AverageMeter] = {}
        self.count = 0

    def feed(self, losses: Dict, batch_size: int = 1, **kwargs):
        for k, v in losses.items():
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                continue

            if k in self._losses:
                self._losses[k].update_by_mean(v.item(), batch_size)
            else:
                self._losses[k] = AverageMeter()
                self._losses[k].update_by_mean(v.item(), batch_size)

        self.count += batch_size

    def get_measures(self, **kwargs) -> Dict:
        measure = {}
        for k, v in self._losses.items():
            measure[k] = v.avg
        return measure

    def get_loss(self, loss_name: str) -> float:
        return self._losses[loss_name].avg

    def __str__(self) -> str:
        if self.count == 0:
            return "loss N/A"
        out = ", ".join(
            [f"final_loss: {self._losses['final_loss']}"]
            + [f"{k}: {v}" for k, v in self._losses.items() if k in self._vis_loss_keys]
        )
        return out
