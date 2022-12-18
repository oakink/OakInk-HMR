import os
import random
from typing import Iterable

import numpy as np
import torch
import transformers
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from oib.utils.logger import logger


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def recurse_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        recurse_freeze(child)


def build_optimizer(params: Iterable, **cfg) -> Optimizer:
    if cfg["OPTIMIZER"] == "Adam" or cfg["OPTIMIZER"] == "adam":
        return torch.optim.Adam(params, lr=cfg["LR"], weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)))

    elif cfg["OPTIMIZER"] == "SGD" or cfg["OPTIMIZER"] == "sgd":
        return torch.optim.SGD(params,
                               lr=cfg["LR"],
                               momentum=float(cfg.get("MOMENTUM", 0.0)),
                               weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)))
    else:
        raise NotImplementedError(f"{cfg['OPTIMIZER']} not yet be implemented")


def build_scheduler(optimizer: Optimizer, **cfg):
    scheduler = cfg.get("SCHEDULER", "StepLR")
    if scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, cfg["LR_DECAY_STEP"], gamma=cfg["LR_DECAY_GAMMA"])

    elif scheduler == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=cfg["LR_DECAY_STEP"],
                                                    gamma=cfg["LR_DECAY_GAMMA"])

    elif scheduler == "constant_warmup":
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg["NUM_WARMUP_STEPS"])

    elif scheduler == "cosine_warmup":
        return transformers.get_cosine_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=cfg["NUM_WARMUP_STEPS"],
                                                            num_training_steps=cfg["NUM_TRAINING_STEPS"])

    elif scheduler == "linear_warmup":
        return transformers.get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=cfg["NUM_WARMUP_STEPS"],
                                                            num_training_steps=cfg["NUM_TRAINING_STEPS"])
    else:
        raise NotImplementedError(f"{scheduler} not yet be implemented")


def clip_gradient(optimizer, max_norm, norm_type):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer (torch.optim.optimizer): optimizer with the gradients to be clipped
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def setup_seed(seed, conv_repeatable=True):
    """Setup all the random seeds

    Args:
        seed (int or float): seed value
        conv_repeatable (bool, optional): Whether the conv ops are repeatable (depend on cudnn). Defaults to True.
    """
    logger.warning(f"setup random seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")
