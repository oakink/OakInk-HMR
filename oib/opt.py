import argparse
import os
import re
from time import mktime, strptime, time

import torch

from .utils.logger import logger
from .utils.misc import update_config

_parser = argparse.ArgumentParser(description="MR. Anderson")

_parser.add_argument("--vis_toc", type=float, default=5)
"----------------------------- Experiment options -----------------------------"
_parser.add_argument("--cfg", help="experiment configure file name", type=str, default=None)
_parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")

_parser.add_argument("--resume", help="resume training from exp", type=str, default=None)
_parser.add_argument("--resume_epoch", help="resume from the given epoch", type=int, default=0)
_parser.add_argument("--reload", help="reload checkpoint for test", type=str, default=None)

_parser.add_argument("--workers", help="worker number from data loader", type=int, default=8)
_parser.add_argument(
    "--batch_size", help="batch size of exp, will replace bs in cfg file if is given", type=int, default=None
)

_parser.add_argument(
    "--val_batch_size",
    help="batch size when val or test, will replace bs in cfg file if is given",
    type=int,
    default=None,
)

_parser.add_argument(
    "--recon_res",
    help="reconstruction resolution of exp, will replace rr in cfg file if is given",
    type=int,
    default=None,
)

_parser.add_argument("--evaluate", help="evaluate the network (ignore training)", action="store_true")
"----------------------------- General options -----------------------------"
_parser.add_argument("--gpu_id", type=str, default=None, help="override enviroment var CUDA_VISIBLE_DEVICES")
_parser.add_argument("--snapshot", default=10, type=int, help="How often to take a snapshot of the model (0 = never)")
_parser.add_argument("--eval_interval", default=5, type=int, help="How often to evaluate the model on val set")
_parser.add_argument("--test_freq", type=int, default=200, help="How often to test, 1 for always -1 for never")
"----------------------------- Distributed options -----------------------------"
_parser.add_argument("--dist_master_addr", type=str, default="localhost")
_parser.add_argument("--dist_master_port", type=str, default="auto")
_parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
_parser.add_argument("--nodes", default=1, type=int, help="nodes number for distributed training")
_parser.add_argument("--dist_backend", default="gloo", type=str, help="distributed backend")
_parser.add_argument("-ddp", "--distributed", default=True, type=bool, help="Use distributed data parallel")
"-------------------------------------dataset submit options-------------------------------------"

_parser.add_argument("--submit_dataset", type=str, default="freiHAND", choices=["HO3D", "freiHAND"])


def parse_exp_args():
    arg, custom_arg_string = _parser.parse_known_args()

    if arg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    arg.n_gpus = torch.cuda.device_count()
    arg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return arg, custom_arg_string


def parse_test_args(arg):
    if arg.reload is None:
        logger.warning("reload checkpoint path is required")
        return arg, time()
    reload_path_list = arg.reload.split("/")
    try:
        exp_id = reload_path_list[-4]
        exp_match = re.compile(r"(.+)_([0-9]{4})_([0-9]{4})_([0-9]{4})_([0-9]{2})")
        exp_list = exp_match.findall(exp_id)[0]
        exp_name = f"_{exp_list[0]}"
        exp_date = "_".join(exp_list[1:])
        t = mktime(strptime(exp_date, "%Y_%m%d_%H%M_%S"))
    except:
        exp_name = ""
        t = time()
    try:
        ckp_id = reload_path_list[-2]
        ckp_match = re.compile(r"checkpoint_([0-9]+)")
        ckp_epoch = ckp_match.findall(ckp_id)[0]
        ckp_epoch = f"_ckp{ckp_epoch}"
    except:
        ckp_epoch = ""
    arg.exp_id = f"eval{exp_name}{ckp_epoch}" if arg.exp_id == "default" else arg.exp_id
    return arg, t
