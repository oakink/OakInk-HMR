from argparse import Namespace
from copy import deepcopy

from yacs.config import CfgNode as CN
from oib.utils.logger import logger


class CN_R(CN):
    def recursive_cfg_update(self):

        for k, v in self.items():
            if isinstance(v, list):
                for i, v_ in enumerate(v):
                    if isinstance(v_, dict):
                        new_v = CN_R(v_, new_allowed=True)
                        v[i] = new_v.recursive_cfg_update()
            elif isinstance(v, CN) or issubclass(type(v), CN):
                new_v = CN_R(v, new_allowed=True)
                self[k] = new_v.recursive_cfg_update()
        self.freeze()
        return self

    def dump(self, *args, **kwargs):
        def change_back(cfg: CN_R) -> dict:
            for k, v in cfg.items():
                if isinstance(v, list):
                    for i, v_ in enumerate(v):
                        if isinstance(v_, CN_R):
                            new_v = change_back(v_)
                            v[i] = new_v
                elif isinstance(v, CN_R):
                    new_v = change_back(v)
                    cfg[k] = new_v
            return dict(cfg)

        cfg = change_back(deepcopy(self))
        return CN(cfg).dump(*args, **kwargs)


_C = CN(new_allowed=True)

_ZERO_AUG_PARAM = CN(new_allowed=True)
_ZERO_AUG_PARAM.SCALE_JIT = 0.0
_ZERO_AUG_PARAM.COLOR_JIT = 0.0
_ZERO_AUG_PARAM.ROT_JIT = 0.0
_ZERO_AUG_PARAM.ROT_PROB = 0.0
_ZERO_AUG_PARAM.OCCLUSION = False
_ZERO_AUG_PARAM.OCCLUSION_PROB = 0.0
_ZERO_AUG_PARAM.freeze()

_DEFAULT_AUG_PARAM = CN(new_allowed=True)
_DEFAULT_AUG_PARAM.SCALE_JIT = 0.125
_DEFAULT_AUG_PARAM.COLOR_JIT = 0.3
_DEFAULT_AUG_PARAM.ROT_JIT = 60
_DEFAULT_AUG_PARAM.ROT_PROB = 1.0
_DEFAULT_AUG_PARAM.OCCLUSION = True
_DEFAULT_AUG_PARAM.OCCLUSION_PROB = 0.3
_DEFAULT_AUG_PARAM.freeze()

_C.DATA_PRESET = CN(new_allowed=True)
_C.DATA_PRESET.USE_CACHE = True
_C.DATA_PRESET.BBOX_EXPAND_RATIO = 1.7
_C.DATA_PRESET.IMAGE_SIZE = (256, 256)
_C.DATA_PRESET.HEATMAP_SIZE = (64, 64)
_C.DATA_PRESET.HEATMAP_SIGMA = 2.0
_C.DATA_PRESET.CENTER_IDX = 9

_C.DATASET = CN(new_allowed=True)
_C.DATASET.TRAIN = CN(new_allowed=True)

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.MANUAL_SEED = 1
_C.TRAIN.CONV_REPEATABLE = True
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCH = 100
_C.TRAIN.OPTIMIZER = "Adam"
_C.TRAIN.LR = 0.001
_C.TRAIN.SCHEDULER = "StepLR"
_C.TRAIN.LR_DECAY_GAMMA = 0.1
_C.TRAIN.LR_DECAY_STEP = [70]
_C.TRAIN.LOG_INTERVAL = 50
_C.TRAIN.FIND_UNUSED_PARAMETERS = False

_C.TRAIN.GRAD_CLIP_ENABLED = True
_C.TRAIN.GRAD_CLIP = CN(new_allowed=True)
_C.TRAIN.GRAD_CLIP.TYPE = 2
_C.TRAIN.GRAD_CLIP.NORM = 0.001

_C.MODEL = CN(new_allowed=True)


def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_config(config_file: str, arg: Namespace = None, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
        cfg = default_config()
    else:
        cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    if arg is not None:
        # if arg.batch_size is given, it always have higher priority
        if arg.batch_size is not None:
            if arg.resume is None:
                logger.warning(f"cfg's batch_size {cfg.TRAIN.BATCH_SIZE} reset to arg.batch_size: {arg.batch_size}")
            cfg.TRAIN.BATCH_SIZE = arg.batch_size
        else:
            arg.batch_size = cfg.TRAIN.BATCH_SIZE

        if arg.recon_res is not None:
            cfg.DATA_PRESET.RECON_RES = arg.recon_res
        else:
            arg.recon_res = cfg.DATA_PRESET.get("RECON_RES", -1)

        # if arg.reload is given, it always have higher priority.
        if arg.reload is not None:
            logger.warning(f"cfg MODEL's pretrained {cfg.MODEL.PRETRAINED} reset to arg.reload: {arg.reload}")
            cfg.MODEL.PRETRAINED = arg.reload

    cfg = CN_R(cfg, new_allowed=True)
    cfg.recursive_cfg_update()
    # cfg.freeze()
    return cfg


if __name__ == "__main__":
    cfg: CN = get_config("config/train_bihand2d_fh_pl.yml")
    print(cfg)
    cfg_str = cfg.dump(sort_keys=False)
    with open("tmp/test_dump_cfg.yaml", "w") as f:
        f.write(cfg_str)
