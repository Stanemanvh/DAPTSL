import logging

import torch
from torch import nn

from args_parser import ArgsParser
from client_uSL import ClientUShapedSL
from u_shaped_dino.DINODataLoaderPartitioner import DINODataLoaderPartitioner
from u_shaped_dino.dinohead import DinoHead
from u_shaped_dino.vithead import DinoVitHead


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
    inputs_dtype = torch.half
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


def main(cfg, args):
    n_clients = getattr(cfg.train, "n_clients", 12)
    clients = build_clients_from_cfg(
        cfg=cfg,
        n_clients=n_clients,
        start_iter=0,
    )

    logger.info(
        "Constructed %d clients",
        len(clients),
    )

    test_clients_forward(clients)


if __name__ == "__main__":
    parser = ArgsParser()
    cfg, args = parser.parse_and_setup()
    main(cfg, args)
