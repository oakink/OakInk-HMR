import inspect

from yacs.config import CfgNode as CN


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, but got {}".format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError("{} is already registered in {}".format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


MODEL = Registry("model")
DATASET = Registry("dataset")
LOSS = Registry("loss")
METRIC = Registry("metric")


def build_from_cfg(cfg, registry):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "TYPE".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, CN) and cfg.TYPE is not None
    obj_type = cfg.TYPE

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError("{} is not in the {} registry".format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError("type must be a str or valid type, but got {}".format(type(obj_type)))

    return obj_cls(cfg)


def build_dataset(cfg: CN, preset_cfg: CN, **kwargs):
    # exec(f"from ..datasets import {cfg.TYPE}")
    cfg = cfg.clone()
    cfg.defrost()

    if preset_cfg is not None:
        cfg.DATA_PRESET = preset_cfg

    cfg.freeze()

    return build_from_cfg(cfg, DATASET)


def build_loss(cfg: CN, preset_cfg: CN, **kwargs):
    exec(f"from ..criterions import {cfg.TYPE}")
    cfg = cfg.clone()
    cfg.defrost()

    if preset_cfg is not None:
        cfg.DATA_PRESET = preset_cfg

    cfg.freeze()

    return build_from_cfg(cfg, LOSS)


def build_metric(cfg: CN, preset_cfg: CN, **kwargs):
    exec(f"from ..metrics import {cfg.TYPE}")
    cfg = cfg.clone()
    cfg.defrost()

    if preset_cfg is not None:
        cfg.DATA_PRESET = preset_cfg

    cfg.freeze()

    return build_from_cfg(cfg, METRIC)


def build_model(cfg: CN, preset_cfg: CN = None, train_cfg: CN = None, **kwargs):
    # exec(f"from ..models import {cfg.TYPE}")

    cfg = cfg.clone()
    cfg.defrost()

    if preset_cfg is not None:
        cfg.DATA_PRESET = preset_cfg

    if train_cfg is not None:
        cfg.TRAIN = train_cfg

    cfg.freeze()

    model = build_from_cfg(cfg, MODEL)
    return model
