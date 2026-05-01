# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from .util import misc as linprobe_utils
from .model_setup import build_model_from_cfg
from .config_setup import setup


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        nargs="+",
        help="Pretrained model weights to load in order (later checkpoints override earlier ones)",
    )
    parser.add_argument(
        "--pretrained-checkpoint-keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional checkpoint keys for --pretrained-weights in the same order "
            "(e.g., model teacher). Use none to load from root state_dict for that checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, pretrained_checkpoint_keys=None):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    default_checkpoint_key = config.student.checkpoint_key if hasattr(config.student, "checkpoint_key") else "teacher"

    # Backward compatibility: accept either a single path or multiple paths.
    if isinstance(pretrained_weights, str):
        pretrained_weights = [pretrained_weights]

    if pretrained_checkpoint_keys is None:
        checkpoint_keys = [default_checkpoint_key] * len(pretrained_weights)
    else:
        if isinstance(pretrained_checkpoint_keys, str):
            pretrained_checkpoint_keys = [pretrained_checkpoint_keys]
        if len(pretrained_checkpoint_keys) != len(pretrained_weights):
            raise ValueError(
                "--pretrained-checkpoint-keys must have the same number of entries as --pretrained-weights"
            )
        checkpoint_keys = [None if str(k).lower() in {"none", "null", ""} else k for k in pretrained_checkpoint_keys]

    for weights_path, checkpoint_key in zip(pretrained_weights, checkpoint_keys):
        linprobe_utils.load_pretrained_weights(
            model,
            weights_path,
            checkpoint_key,
        )
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights, args.pretrained_checkpoint_keys)
    autocast_dtype = get_autocast_dtype(config)
    return model, config, autocast_dtype
