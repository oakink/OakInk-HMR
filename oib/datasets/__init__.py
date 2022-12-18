from oib.utils.builder import build_dataset
from yacs.config import CfgNode as CN

from .mix_dataset import MixDataset
from .oakink import OakInk


def create_dataset(cfg: CN, preset_cfg: CN, **kwargs):
    """
    Create a dataset instance.
    """
    if cfg.TYPE == "MixDataset":
        # list of CN of each dataset
        dataset_list = [v for k, v in cfg.DATASET_LIST.items()]
        return MixDataset(dataset_list, preset_cfg, cfg.get("MAX_LEN"))
    else:
        # default building from cfg
        return build_dataset(cfg, preset_cfg, **kwargs)
