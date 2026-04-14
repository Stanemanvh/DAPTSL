# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path
import copy


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util.timm import trunc_normal_, Mixup, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
import models_vit_temporal
import models_vit_group_channels
from train_lora_util import activate_lora, deactivate_lora
from vit_lora_util import Block as LoraBlock


from engine_finetune import train_one_epoch, train_one_epoch_temporal, evaluate, evaluate_temporal


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        default=None,
        choices=["group_c", "resnet", "resnet_pre", "temporal", "vanilla"],
        help="Use channel model",
    )
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train"
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--patch_size", default=16, type=int, help="images input size")

    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")

    # LoRA info
    parser.add_argument(
        "--lora_type",
        type=str,
        default="lora",
        choices=["lora", "boft", "monarch"],
        help="Which matrix decomposition type to use.",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="Lora matrices rank")
    parser.add_argument("--lora_layers", type=str, default=["attn"], nargs="+", help="Layers to use lora on")
    parser.add_argument(
        "--unfreeze_blocks", type=int, default=None, nargs="+", help="Which ViT blocks to unfreeze to train fully."
    )
    parser.add_argument(
        "--unfreeze_embed", action="store_true", default=False, help="Whether to train patch embed. Default is false."
    )
    parser.add_argument(
        "--unfreeze_cls_token",
        action="store_true",
        default=False,
        help="Whether to train cls_token. Default is false.",
    )
    parser.add_argument(
        "--unfreeze_norm", action="store_true", default=False, help="Whether to train norms. Default is false."
    )
    parser.add_argument(
        "--lora_attn_key", action="store_true", default=False, help="Also use lora on key proj weights."
    )
    parser.add_argument(
        "--lora_attn_proj",
        action="store_true",
        default=False,
        help="Also use lora on attn proj weights (used after qkv).",
    )
    parser.add_argument("--unfreeze_all", default=False, action="store_true")

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (default: 0.0)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--layer_decay", type=float, default=1.0, help="layer-wise lr decay from ELECTRA/BEiT")

    parser.add_argument(
        "--min_lr", type=float, default=1e-6, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR")

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument(
        "--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split"
    )

    # * Mixup params
    parser.add_argument("--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0.")
    parser.add_argument("--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0.")
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument(
        "--train_path", default="data_and_checkpoints/fmow_csvs/train_62classes.csv", type=str, help="Train .csv path"
    )
    parser.add_argument(
        "--test_path", default="data_and_checkpoints/fmow_csvs/test_62classes.csv", type=str, help="Test .csv path"
    )
    parser.add_argument(
        "--dataset_type",
        default="rgb",
        choices=[
            "rgb",
            "temporal",
            "sentinel",
            "euro_sat",
            "naip",
            "resisc45",
            "camelyon17",
            "globalwheat",
            "iwildcam",
            "poverty",
            "rxrx1",
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        default=None,
        nargs="+",
        type=int,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )
    parser.add_argument(
        "--grouped_bands", type=int, nargs="+", action="append", default=[], help="Bands to group for GroupC vit"
    )

    parser.add_argument("--nb_classes", default=62, type=int, help="number of the classification types")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=1, help="How frequently (in epochs) to save ckpt")
    parser.add_argument("--wandb", type=str, default=None, help="Wandb project name, eg: sentinel_finetune")
    parser.add_argument(
        "--wandb_logdir",
        type=str,
        default="./wandb",
        help="Where the wandb log info is stored. Warning: takes a lot of space fast",
    )
    parser.add_argument("--wandb_entity", type=str, default="FILL IN", help="Wandb entity name")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_fmow_dataset(is_train=True, args=args)
    dataset_val = build_fmow_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=dataset_train.n_classes,
        )

    # Define block type
    block_type = LoraBlock

    # Define the model
    if args.model_type == "group_c":
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_vit_group_channels.__dict__[args.model](
            block_type=block_type,
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands,
            num_classes=dataset_train.n_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        # Dino group channels for eval
        if args.finetune is not None and "dino" in args.finetune:
            model = models_vit_dino_group_channels.__dict__[args.model](
                block_type=block_type,
                patch_size=args.patch_size,
                img_size=args.input_size,
                in_chans=dataset_train.in_c,
                channel_groups=args.grouped_bands,
                num_classes=dataset_train.n_classes,
                drop_path_rate=args.drop_path,
                global_pool=args.global_pool,
            )
    elif args.model_type == "temporal":
        model = models_vit_temporal.__dict__[args.model](
            block_type=block_type,
            num_classes=dataset_train.n_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        # NOTE: Make sure DINO pretrained checkpoints have "dino" in the path to use layerscale
        model = models_vit.__dict__[args.model](
            block_type=block_type,
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            num_classes=dataset_train.n_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            use_ls=("dino" in args.finetune) if args.finetune is not None else False,
        )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                if k.startswith("student.backbone"):
                    new_checkpoint_model[k.replace("student.backbone.", "")] = copy.deepcopy(v)
                elif k.startswith("encoder"):
                    new_checkpoint_model[k.replace("encoder.", "")] = copy.deepcopy(v)
                else:
                    new_checkpoint_model[k] = copy.deepcopy(v)
            del checkpoint_model
            checkpoint_model = new_checkpoint_model
        elif "teacher" in checkpoint:
            checkpoint_model = checkpoint["teacher"]
            new_checkpoint_model = {}
            for k, v in checkpoint_model.items():
                new_checkpoint_model[k.replace("backbone.", "")] = copy.deepcopy(v)
            del checkpoint_model
            checkpoint_model = new_checkpoint_model
        else:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()

        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]

        # TODO: Do something smarter?
        # Removed pos_embed from this list, but add back if going from Group_C -> RGB or smthg
        for k in [
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "head.weight",
            "head.bias",
            "patch_embed.0.proj.weight",
            "patch_embed.0.proj.bias",
            "patch_embed.1.proj.weight",
            "patch_embed.1.proj.bias",
            "patch_embed.2.proj.weight",
            "patch_embed.2.proj.bias",
            "decoder_pred.0.weight",
            "decoder_pred.0.bias",
            "decoder_pred.1.weight",
            "decoder_pred.1.bias",
            "decoder_pred.2.weight",
            "decoder_pred.2.bias",
        ]:
            if k not in state_dict and k not in checkpoint_model:
                continue

            if k not in state_dict or (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # Set up lora
        if not args.unfreeze_all:
            if args.unfreeze_blocks is not None:
                lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in args.unfreeze_blocks]
                for block_idx in lora_blocks:
                    activate_lora(
                        model.blocks[block_idx],
                        args.lora_layers,
                        args.lora_rank,
                        include_attn_key=args.lora_attn_key,
                        include_attn_proj=args.lora_attn_proj,
                    )
            else:
                activate_lora(
                    model.blocks,
                    args.lora_layers,
                    args.lora_rank,
                    include_attn_key=args.lora_attn_key,
                    include_attn_proj=args.lora_attn_proj,
                )

            for name, param in model.named_parameters():
                if not (
                    "head" in name
                    or args.lora_type in name
                    or (args.unfreeze_embed and ("patch_embed" in name or "decoder_pred" in name))
                    or (args.unfreeze_norm and "norm" in name)
                    or (args.unfreeze_cls_token and "cls_token" in name)
                    or (
                        args.unfreeze_blocks is not None
                        and any([f"blocks.{idx}." in name and "decoder" not in name for idx in args.unfreeze_blocks])
                    )
                ):
                    param.requires_grad_(False)

        # TODO: change assert msg based on patch_embed
        if args.global_pool:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.4f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if args.model_type is not None and args.model_type.startswith("resnet"):
        param_groups = model_without_ddp.parameters()
    else:
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, dir=args.wandb_logdir, entity=args.wandb_entity)
        wandb.config.update(args)
        wandb.watch(model)

    if args.eval:
        if args.model_type == "temporal":
            test_stats = evaluate_temporal(data_loader_val, model, device)
        else:
            test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
            f"acc5: {test_stats['acc5']:.2f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.model_type == "temporal":
            train_stats = train_one_epoch_temporal(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                mixup_fn,
                log_writer=log_writer,
                args=args,
            )
        else:
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                mixup_fn,
                log_writer=log_writer,
                args=args,
            )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        if args.model_type == "temporal":
            test_stats = evaluate_temporal(data_loader_val, model, device)
        else:
            test_stats = evaluate(data_loader_val, model, device, use_top5=dataset_val.n_classes >= 5)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
