# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn

from u_shaped_dino.vitbackbone import DinoViTBackbone


logger = logging.getLogger("dinov2")


def _arch_to_vit_kwargs(arch_name: str):
    arch = arch_name.removesuffix("_memeff")
    arch_to_kwargs = {
        "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "vit_giant2": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
        "vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "vitg14": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
    }
    if arch not in arch_to_kwargs:
        raise ValueError(f"Unsupported architecture: {arch}")
    return arch_to_kwargs[arch]


def build_model_from_cfg(cfg):
    student_cfg = cfg.student
    base_kwargs = _arch_to_vit_kwargs(student_cfg.arch)
    base_kwargs.update(
        {
            "patch_size": student_cfg.patch_size,
            "mlp_ratio": getattr(student_cfg, "mlp_ratio", 4.0),
            "qkv_bias": getattr(student_cfg, "qkv_bias", True),
            "proj_bias": getattr(student_cfg, "proj_bias", True),
            "ffn_bias": getattr(student_cfg, "ffn_bias", True),
            "ffn_layer": getattr(student_cfg, "ffn_layer", "mlp"),
            "block_chunks": getattr(student_cfg, "block_chunks", 1),
            "num_register_tokens": getattr(student_cfg, "num_register_tokens", 0),
            "init_values": getattr(student_cfg, "layerscale", None),
        }
    )

    teacher = DinoViTBackbone(**base_kwargs)
    student = DinoViTBackbone(
        **base_kwargs,
        drop_path_rate=getattr(student_cfg, "drop_path_rate", 0.0),
        drop_path_uniform=getattr(student_cfg, "drop_path_uniform", False),
    )
    return student, teacher


class DinoBackbone(nn.Module):
    """
    Server-side student/teacher ViT backbone.

    Input is the client-side tokenized tuple:
    (global_unmasked_tokens, global_masked_tokens, local_tokens, masks)
    """

    def __init__(self, cfg):
        super().__init__()
        self.student_backbone, self.teacher_backbone = build_model_from_cfg(cfg)

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

    def forward(self, client_embeddings):
        if isinstance(client_embeddings, dict):
            global_unmasked_tokens = client_embeddings["global_unmasked"]
            global_masked_tokens = client_embeddings["global_masked"]
            local_tokens = client_embeddings["local_unmasked"]
            masks = client_embeddings["masks"]
        else:
            global_unmasked_tokens, global_masked_tokens, local_tokens, masks = client_embeddings

        with torch.no_grad():
            teacher_global_out = self.teacher_backbone(global_unmasked_tokens, masks=None)

        student_global_out = self.student_backbone(global_masked_tokens, masks=masks)
        student_local_out = self.student_backbone(local_tokens, masks=None)

        # Flat fields are convenient for loss computations downstream.
        return {
            "teacher": teacher_global_out,
            "student_global": student_global_out,
            "student_local": student_local_out,
            "masks": masks,
            "teacher_cls_tokens": teacher_global_out["x_norm_clstoken"],
            "teacher_patch_tokens": teacher_global_out["x_norm_patchtokens"],
            "student_global_cls_tokens": student_global_out["x_norm_clstoken"],
            "student_global_patch_tokens": student_global_out["x_norm_patchtokens"],
            "student_local_cls_tokens": student_local_out["x_norm_clstoken"],
        }
    


