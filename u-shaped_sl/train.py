import logging
import math

import torch
from torch import nn

from args_parser import ArgsParser
from client_uSL import ClientUShapedSL
from server_uSL import ServerUShapedSL
from u_shaped_dino.DINODataLoaderPartitioner import DINODataLoaderPartitioner
from u_shaped_dino.dinobackbone import DinoBackbone
from u_shaped_dino.dino_loss import DinoLoss
from u_shaped_dino.dinohead import DinoHead
from u_shaped_dino.vithead import DinoVitHead
from dinov2.utils.utils import CosineScheduler


logger = logging.getLogger("u_shaped_sl")


def _embed_dim_from_arch(arch_name: str) -> int:
    arch = arch_name.removesuffix("_memeff")
    arch_to_dim = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_giant2": 1536,
        "vits14": 384,
        "vitb14": 768,
        "vitl14": 1024,
        "vitg14": 1536,
    }
    if arch not in arch_to_dim:
        raise ValueError(f"Unsupported architecture for DinoVitHead: {arch}")
    return arch_to_dim[arch]


def build_vithead_from_cfg(cfg):
    student_cfg = cfg.student
    return DinoVitHead(
        img_size=cfg.crops.global_crops_size,
        patch_size=student_cfg.patch_size,
        in_chans=getattr(student_cfg, "in_chans", 3),
        embed_dim=_embed_dim_from_arch(student_cfg.arch),
        num_register_tokens=getattr(student_cfg, "num_register_tokens", 0),
        interpolate_antialias=getattr(student_cfg, "interpolate_antialias", False),
        interpolate_offset=getattr(student_cfg, "interpolate_offset", 0.1),
    )


def build_clients_from_cfg(cfg, n_clients: int = 12, start_iter: int = 0):
    inputs_dtype = torch.float32
    partitioner = DINODataLoaderPartitioner(
        dataset_path=cfg.train.dataset_path,
        global_crops_size=cfg.crops.global_crops_size,
        patch_size=cfg.student.patch_size,
        mask_ratio_min_max=cfg.ibot.mask_ratio_min_max,
        mask_sample_probability=cfg.ibot.mask_sample_probability,
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        local_crops_number=cfg.crops.local_crops_number,
        local_crops_size=cfg.crops.local_crops_size,
        batch_size_per_gpu=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        inputs_dtype=inputs_dtype,
    )

    dataloaders = partitioner.get_partitioned_dataloaders(n_partitions=n_clients, start_iter=start_iter)

    clients = []

    for idx in range(n_clients):
        vit_head = build_vithead_from_cfg(cfg).to(torch.device("cuda"))
        dino_head = DinoHead(vit_head).to(torch.device("cuda"))
        tail = nn.Identity().to(torch.device("cuda"))
        client = ClientUShapedSL(head=dino_head, tail=tail, DataLoader=dataloaders[idx])

        clients.append(client)

    return clients


def _format_output_summary(output):
    if isinstance(output, dict):
        summary = {}
        for key, value in output.items():
            if torch.is_tensor(value):
                summary[key] = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
            elif isinstance(value, dict):
                summary[key] = f"dict(keys={list(value.keys())})"
            else:
                summary[key] = type(value).__name__
        return repr(summary)
    if isinstance(output, tuple):
        summary_parts = []
        for item in output:
            if torch.is_tensor(item):
                summary_parts.append(f"Tensor(shape={tuple(item.shape)}, dtype={item.dtype})")
            else:
                summary_parts.append(type(item).__name__)
        return "(" + ", ".join(summary_parts) + ")"
    if torch.is_tensor(output):
        return f"Tensor(shape={tuple(output.shape)}, dtype={output.dtype})"
    return repr(output)


def test_clients_forward(clients):
    for idx, client in enumerate(clients):
        output = client.forwardHead()
        print(f"client[{idx}] -> {_format_output_summary(output)}")


def test_clients_and_server_forward(clients, server):
    for idx, client in enumerate(clients):
        client_output = client.forwardHead()
        backbone_output = server.forwardBackbone(client_output)
        print(f"client[{idx}] -> {_format_output_summary(client_output)}")
        print(f"backbone[{idx}] -> {_format_output_summary(backbone_output)}")


def _aggregate_client_outputs(client_outputs):
    global_unmasked = torch.cat([out[0] for out in client_outputs], dim=0)
    global_masked = torch.cat([out[1] for out in client_outputs], dim=0)
    local_unmasked = torch.cat([out[2] for out in client_outputs], dim=0)
    masks = torch.cat([out[3] for out in client_outputs], dim=0)
    return global_unmasked, global_masked, local_unmasked, masks


def _build_optimizer(cfg, clients, server):
    trainable_params = []
    for client in clients:
        trainable_params.extend(client.head.parameters())
    trainable_params.extend(server.backbone.student_parameters())

    return torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        weight_decay=cfg.optim.weight_decay,
    )


def _build_momentum_schedule(cfg):
    official_epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    return CosineScheduler(
        base_value=cfg.teacher.momentum_teacher,
        final_value=cfg.teacher.final_momentum_teacher,
        total_iters=cfg.optim.epochs * official_epoch_length,
    )


def _build_teacher_temp_schedule(cfg):
    official_epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    warmup_iters = cfg.teacher.warmup_teacher_temp_epochs * official_epoch_length
    return CosineScheduler(
        base_value=cfg.teacher.teacher_temp,
        final_value=cfg.teacher.teacher_temp,
        total_iters=max(warmup_iters, 1),
        warmup_iters=warmup_iters,
        start_warmup_value=cfg.teacher.warmup_teacher_temp,
    )


def _current_teacher_temp(teacher_temp_schedule, iteration):
    if iteration < len(teacher_temp_schedule.schedule):
        return teacher_temp_schedule[iteration]
    return teacher_temp_schedule.schedule[-1]


def do_train(cfg, clients, server, start_iter: int = 0):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    log_period = getattr(cfg.train, "log_period", 10)
    accum_iter = getattr(cfg.optim, "accum_iter", 1)

    optimizer = _build_optimizer(cfg, clients, server)
    momentum_schedule = _build_momentum_schedule(cfg)
    teacher_temp_schedule = _build_teacher_temp_schedule(cfg)
    all_trainable_params = [p for group in optimizer.param_groups for p in group["params"]]

    for client in clients:
        client.head.train()
        client.tail.train()
    server.backbone.train()

    iteration = start_iter
    logger.info("Starting training loop from iteration %d to %d", start_iter, max_iter)

    while iteration < max_iter:
        if iteration % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        teacher_temp = _current_teacher_temp(teacher_temp_schedule, iteration)

        client_outputs = [client.forwardHead() for client in clients]
        aggregated_output = _aggregate_client_outputs(client_outputs)
        backbone_output, loss_dict = server.forwardBackboneAndComputeLoss(
            aggregated_output,
            teacher_temp=teacher_temp,
        )

        server.backwardLoss(loss_dict=loss_dict, accum_iter=accum_iter)

        if (iteration + 1) % accum_iter == 0:
            if getattr(cfg.optim, "clip_grad", 0):
                torch.nn.utils.clip_grad_norm_(all_trainable_params, cfg.optim.clip_grad)

            optimizer.step()

            momentum = momentum_schedule[iteration]
            server.backbone.update_teacher(momentum)

        reduced_loss = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                reduced_loss[k] = float(v.detach().item())
            else:
                reduced_loss[k] = v

        if math.isnan(reduced_loss["total_loss"]):
            raise RuntimeError(f"NaN detected in total_loss at iteration {iteration}")

        if (iteration + 1) % log_period == 0 or iteration == start_iter:
            logger.info("Iteration %d / %d", iteration, max_iter)
            logger.info("Backbone output summary: %s", _format_output_summary(backbone_output))
            logger.info("Loss summary: %s", reduced_loss)
            logger.info("Teacher temperature: %.6f", teacher_temp)

        iteration = iteration + 1

    logger.info("Training loop complete at iteration %d", max_iter)


def main(cfg, args):
    n_clients = getattr(cfg.train, "n_clients", 4)
    device = torch.device("cuda")
    clients = build_clients_from_cfg(
        cfg=cfg,
        n_clients=n_clients,
        start_iter=0,
    )
    server = ServerUShapedSL(
        backbone=DinoBackbone(cfg).to(device),
        dino_loss=DinoLoss(cfg).to(device),
    )

    logger.info(
        "Constructed %d clients",
        len(clients),
    )

    do_train(cfg, clients, server, start_iter=0)


if __name__ == "__main__":
    parser = ArgsParser()
    cfg, args = parser.parse_and_setup()
    main(cfg, args)
