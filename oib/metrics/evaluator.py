from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL.Image import Image

from ..utils.logger import logger
from .basic_metric import AverageMeter, LossMetric, Metric, VisMetric


class Evaluator:
    def __init__(self, metrics_list: List[Metric]) -> None:
        super(Evaluator, self).__init__()
        self._metrics_list = metrics_list
        self.clean = True

    @property
    def metrics_list(self) -> List[Metric]:
        return self._metrics_list

    def reset_all(self):
        for metric in self._metrics_list:
            metric.reset()
        self.clean = True

    @property
    def losses_metric(self) -> Optional[LossMetric]:
        for metric in self._metrics_list:
            if isinstance(metric, LossMetric):
                return metric
        logger.error("No LossMetric found! Please check the config file")
        return None

    def feed_all(self, preds: Dict, targs: Dict, losses: Dict, **kwargs):
        self.clean = False
        batch_size = targs[next(iter(targs))].shape[0]

        for metric in self._metrics_list:
            if isinstance(metric, LossMetric):  # feed losses info to LossMetric
                if len(losses):
                    metric.feed(losses=losses, batch_size=batch_size)
            else:
                metric.feed(preds=preds, targs=targs, **kwargs)

    def get_measures_all(self) -> Dict[str, Dict]:
        measures_all: Dict[str, Dict] = dict()
        for metric in self._metrics_list:
            if isinstance(metric, VisMetric):
                continue
            name = type(metric).__name__
            result_dict = metric.get_measures()

            if measures_all.get(name) is not None:
                logger.warning(f"duplicate metric {name} found! its value will be rewrite !")
            measures_all[f"{name}"] = result_dict
        return measures_all

    def get_measures_all_striped(self, return_losses=True) -> Dict[str, Dict[str, float]]:
        measures_all: Dict[str, Dict] = dict()
        for metric in self._metrics_list:
            if isinstance(metric, VisMetric):
                continue
            if not return_losses and isinstance(metric, LossMetric):
                continue
            name = type(metric).__name__
            result_dict = metric.get_measures()
            if measures_all.get(name) is not None:
                logger.warning(f"duplicate metric {name} found! its value will be rewrite !")
            striped_dict: Dict[str, float] = {}
            for measure_name, measure_content in result_dict.items():
                if isinstance(measure_content, (float, int, np.float64, np.float32, np.int64, np.int32)):
                    striped_dict[measure_name] = float(measure_content)
            measures_all[f"{name}"] = striped_dict
        return measures_all

    def get_metric_images(self) -> Dict[str, Image]:
        images = {}
        for metric in self._metrics_list:
            if not isinstance(metric, VisMetric):
                continue
            if metric.images is None:
                continue
            images[type(metric).__name__] = metric.images
        return images

    def __str__(self) -> str:
        if self.clean:
            return "No metric feeded"
        return " | ".join([str(m_list) for m_list in self._metrics_list if not isinstance(m_list, VisMetric)])
