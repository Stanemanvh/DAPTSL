# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss, iBOTPatchLoss
from u_shaped_dino.vitbackbone import DinoViTBackbone
from dinov2.utils.train_lora_util import is_merged, activate_lora, deactivate_lora, delete_qkv

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
        self.cfg = cfg
        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head
        self.n_global_crops = 2
        self.centering_mode = cfg.train.centering
        self.dino_out_dim = cfg.dino.head_n_prototypes
        self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else self.dino_out_dim

        self.student_backbone, self.teacher_backbone = build_model_from_cfg(cfg)

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")

            # SatMAE style loading weights
            load_chkpt = chkpt["model"] if "model" in chkpt else chkpt
            student_backbone_sd = self.student_backbone.state_dict()
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
            ]:
                if k not in student_backbone_sd and k not in load_chkpt:
                    continue

                if k not in student_backbone_sd or (
                    k in load_chkpt and load_chkpt[k].shape != student_backbone_sd[k].shape
                ):
                    logger.info(f"Removing key {k} from pretrained checkpoint")
                    del load_chkpt[k]

            msg = self.student_backbone.load_state_dict(load_chkpt, strict=False)
            print(msg)
        
        use_lora = hasattr(cfg, "lora")
        if use_lora:
            if cfg.lora.rank > 0:
                self.activate_lora_for_model(
                    self.student_backbone,
                    cfg.lora.unfreeze_blocks,
                    cfg.lora.layers,
                    cfg.lora.rank,
                    cfg.lora.attn_key,
                    cfg.lora.attn_proj,
                )
                self.activate_lora_for_model(
                    self.teacher_backbone,
                    cfg.lora.unfreeze_blocks,
                    cfg.lora.layers,
                    cfg.lora.rank,
                    cfg.lora.attn_key,
                    cfg.lora.attn_proj,
                )

            for name, param in self.student["backbone"].named_parameters():
                if not (
                    "lora" in name
                    or "register_token" in name
                    or (cfg.lora.unfreeze_embed and ("patch_embed" in name or "decoder_pred" in name))
                    or (getattr(cfg.lora, "unfreeze_channel_embed", False) and "channel_embed" in name)
                    or (cfg.lora.unfreeze_norm and "norm" in name)
                    or (cfg.lora.unfreeze_cls_token and "cls_token" in name)
                    or (getattr(cfg.lora, "unfreeze_mask_token", False) and "mask_token" in name)
                    or (
                        cfg.lora.unfreeze_blocks is not None
                        and any([f"blocks.{idx}." in name for idx in cfg.lora.unfreeze_blocks])
                    )
                ):
                    param.requires_grad_(False)

            # there is no backpropagation through the teacher, so no need for gradients (reset this after activating LoRA)
            for p in self.teacher.backbone.parameters():
                p.requires_grad = False

            student_n_params = sum(p.numel() for p in self.student["backbone"].parameters() if p.requires_grad)
            logger.info(f"NUMBER OF TRAINABLE PARAMS: {student_n_params}")

            for k, v in self.student.items():
                self.teacher[k].load_state_dict(self.student[k].state_dict())

        embed_dim = self.student_backbone.embed_dim
        self.student_dino_head = DINOHead(
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        self.teacher_dino_head = DINOHead(
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )

        if self.do_ibot and self.ibot_separate_head:
            self.student_ibot_head = DINOHead(
                in_dim=embed_dim,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=cfg.ibot.head_hidden_dim,
                bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                nlayers=cfg.ibot.head_nlayers,
            )
            self.teacher_ibot_head = DINOHead(
                in_dim=embed_dim,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=cfg.ibot.head_hidden_dim,
                bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                nlayers=cfg.ibot.head_nlayers,
            )
        else:
            self.student_ibot_head = None
            self.teacher_ibot_head = None

        self._sync_teacher_modules()

        self.dino_target_builder = DINOLoss(self.dino_out_dim)
        self.ibot_target_builder = iBOTPatchLoss(self.ibot_out_dim) if self.do_ibot else None

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False
        if self.teacher_ibot_head is not None:
            for p in self.teacher_ibot_head.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _sync_teacher_modules(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        if self.teacher_ibot_head is not None:
            self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())

    def student_parameters(self):
        for p in self.student_backbone.parameters():
            yield p
        for p in self.student_dino_head.parameters():
            yield p
        if self.student_ibot_head is not None:
            for p in self.student_ibot_head.parameters():
                yield p

    @torch.no_grad()
    def update_teacher(self, momentum):
        student_modules = [self.student_backbone, self.student_dino_head]
        teacher_modules = [self.teacher_backbone, self.teacher_dino_head]

        if self.student_ibot_head is not None and self.teacher_ibot_head is not None:
            student_modules.append(self.student_ibot_head)
            teacher_modules.append(self.teacher_ibot_head)

        for student_module, teacher_module in zip(student_modules, teacher_modules):
            for student_param, teacher_param in zip(student_module.parameters(), teacher_module.parameters()):
                teacher_param.mul_(momentum).add_(student_param, alpha=1.0 - momentum)

    def _unpack_client_embeddings(self, client_embeddings):
        if isinstance(client_embeddings, dict):
            global_unmasked_tokens = client_embeddings["global_unmasked"]
            global_masked_tokens = client_embeddings["global_masked"]
            local_tokens = client_embeddings["local_unmasked"]
            masks = client_embeddings["masks"]
        else:
            global_unmasked_tokens, global_masked_tokens, local_tokens, masks = client_embeddings

        return global_unmasked_tokens, global_masked_tokens, local_tokens, masks

    def forward_teacher(self, global_unmasked_tokens):
        with torch.no_grad():
            return self.teacher_backbone(global_unmasked_tokens, masks=None)

    def forward_student_global(self, global_masked_tokens, masks):
        return self.student_backbone(global_masked_tokens, masks=masks)

    def forward_student_local(self, local_tokens):
        return self.student_backbone(local_tokens, masks=None)

    def extract_teacher_tokens(self, teacher_global_out):
        teacher_cls_tokens = self._chunk_teacher_targets(teacher_global_out["x_norm_clstoken"])
        teacher_patch_tokens = teacher_global_out["x_norm_patchtokens"]
        return teacher_cls_tokens, teacher_patch_tokens

    def _select_masked_patch_tokens(self, patch_tokens, masks):
        return patch_tokens[masks.bool()]

    def project_teacher_tokens_with_head(self, teacher_cls_tokens, teacher_patch_tokens, masks):
        teacher_cls_tokens_after_head = self.teacher_dino_head(teacher_cls_tokens)

        masked_teacher_patch_tokens_after_head = None
        if self.do_ibot:
            if self.ibot_separate_head:
                teacher_patch_tokens_after_head = self.teacher_ibot_head(teacher_patch_tokens)
            else:
                teacher_patch_tokens_after_head = self.teacher_dino_head(teacher_patch_tokens)
            masked_teacher_patch_tokens_after_head = self._select_masked_patch_tokens(teacher_patch_tokens_after_head, masks)

        return teacher_cls_tokens_after_head, masked_teacher_patch_tokens_after_head

    @torch.no_grad()
    def center_teacher_outputs(
        self,
        teacher_cls_tokens_after_head,
        masked_teacher_patch_tokens_after_head,
        teacher_temp,
    ):
        if self.centering_mode != "sinkhorn_knopp":
            raise NotImplementedError(
                f"Unsupported centering mode: {self.centering_mode}. Only sinkhorn_knopp is supported."
            )

        teacher_dino_softmaxed_centered_list = self.dino_target_builder.sinkhorn_knopp_teacher(
            teacher_cls_tokens_after_head,
            teacher_temp=teacher_temp,
        ).view(self.n_global_crops, -1, teacher_cls_tokens_after_head.shape[-1])

        masked_teacher_ibot_softmaxed_centered = None
        if self.do_ibot:
            if dist.is_initialized():
                n_masked_patches_tensor = torch.tensor(
                    [masked_teacher_patch_tokens_after_head.shape[0]],
                    device=masked_teacher_patch_tokens_after_head.device,
                    dtype=torch.long,
                )
                masked_teacher_ibot_softmaxed_centered = self.ibot_target_builder.sinkhorn_knopp_teacher(
                    masked_teacher_patch_tokens_after_head,
                    teacher_temp=teacher_temp,
                    n_masked_patches_tensor=n_masked_patches_tensor,
                )
            else:
                masked_teacher_ibot_softmaxed_centered = self.ibot_target_builder.softmax_center_teacher(
                    masked_teacher_patch_tokens_after_head.unsqueeze(0),
                    teacher_temp=teacher_temp,
                ).squeeze(0)
                self.ibot_target_builder.update_center(masked_teacher_patch_tokens_after_head.unsqueeze(0))

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

    @torch.no_grad()
    def get_teacher_output(self, global_unmasked_tokens, masks, teacher_temp):
        teacher_global_out = self.forward_teacher(global_unmasked_tokens)
        teacher_cls_tokens, teacher_patch_tokens = self.extract_teacher_tokens(teacher_global_out)
        teacher_cls_tokens_after_head, masked_teacher_patch_tokens_after_head = self.project_teacher_tokens_with_head(
            teacher_cls_tokens,
            teacher_patch_tokens,
            masks,
        )
        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = self.center_teacher_outputs(
            teacher_cls_tokens_after_head,
            masked_teacher_patch_tokens_after_head,
            teacher_temp,
        )
        return {
            "teacher_dino_softmaxed_centered_list": teacher_dino_softmaxed_centered_list,
            "masked_teacher_ibot_softmaxed_centered": masked_teacher_ibot_softmaxed_centered,
        }

    def _chunk_teacher_targets(self, teacher_cls_tokens):
        if teacher_cls_tokens.shape[0] % self.n_global_crops != 0:
            raise ValueError(
                "teacher cls token batch size must be divisible by n_global_crops; "
                f"got {teacher_cls_tokens.shape[0]} and n_global_crops={self.n_global_crops}."
            )
        teacher_chunks = teacher_cls_tokens.chunk(self.n_global_crops)
        return torch.cat((teacher_chunks[1], teacher_chunks[0]))

    def forward_student_heads(self, student_global_out, student_local_out):
        student_local_cls_tokens = student_local_out["x_norm_clstoken"]
        student_global_cls_tokens = student_global_out["x_norm_clstoken"]

        student_local_cls_tokens_after_head = self.student_dino_head(student_local_cls_tokens)
        student_global_cls_tokens_after_head = self.student_dino_head(student_global_cls_tokens)

        student_global_masked_patch_tokens_after_head = None
        if self.do_ibot:
            student_patch_tokens = student_global_out["x_norm_patchtokens"]
            if self.ibot_separate_head:
                student_patch_tokens_after_head = self.student_ibot_head(student_patch_tokens)
            else:
                student_patch_tokens_after_head = self.student_dino_head(student_patch_tokens)
            student_global_masked_patch_tokens_after_head = self._select_masked_patch_tokens(
                student_patch_tokens_after_head,
                student_global_out["masks"],
            )

        return {
            "student_local_cls_tokens_after_head": student_local_cls_tokens_after_head,
            "student_global_cls_tokens_after_head": student_global_cls_tokens_after_head,
            "student_global_masked_patch_tokens_after_head": student_global_masked_patch_tokens_after_head,
            "student_global_cls_tokens": student_global_cls_tokens if self.do_koleo else None,
        }

    def build_loss_inputs(self, teacher_outputs, student_head_outputs, masks):
        n_masked_patches = int(masks.sum().item())
        return {
            "teacher_dino_softmaxed_centered_list": teacher_outputs["teacher_dino_softmaxed_centered_list"],
            "masked_teacher_ibot_softmaxed_centered": teacher_outputs["masked_teacher_ibot_softmaxed_centered"],
            "student_local_cls_tokens_after_head": student_head_outputs["student_local_cls_tokens_after_head"],
            "student_global_cls_tokens_after_head": student_head_outputs["student_global_cls_tokens_after_head"],
            "student_global_masked_patch_tokens_after_head": student_head_outputs[
                "student_global_masked_patch_tokens_after_head"
            ],
            "student_global_cls_tokens": student_head_outputs["student_global_cls_tokens"],
            "student_masks": masks,
            "n_masked_patches": n_masked_patches,
        }

    def forward_features(self, client_embeddings, teacher_temp=None):
        global_unmasked_tokens, global_masked_tokens, local_tokens, masks = self._unpack_client_embeddings(client_embeddings)

        if teacher_temp is None:
            teacher_temp = getattr(self.cfg.teacher, "teacher_temp", 0.04)

        teacher_outputs = self.get_teacher_output(
            global_unmasked_tokens=global_unmasked_tokens,
            masks=masks,
            teacher_temp=teacher_temp,
        )
        student_global_out = self.forward_student_global(global_masked_tokens, masks)
        student_local_out = self.forward_student_local(local_tokens)

        student_head_outputs = self.forward_student_heads(student_global_out, student_local_out)

        loss_inputs = self.build_loss_inputs(
            teacher_outputs=teacher_outputs,
            student_head_outputs=student_head_outputs,
            masks=masks,
        )

        return {
            "teacher_targets": {
                "teacher_dino_softmaxed_centered_list": teacher_outputs["teacher_dino_softmaxed_centered_list"],
                "masked_teacher_ibot_softmaxed_centered": teacher_outputs["masked_teacher_ibot_softmaxed_centered"],
            },
            "student_heads": student_head_outputs,
            "masks": masks,
            "loss_inputs": loss_inputs,
        }

    def forward(self, client_embeddings):
        return self.forward_features(client_embeddings)
    
    def activate_lora_for_model(model, unfreeze_blocks, lora_layers, lora_rank, lora_attn_key, lora_attn_proj):
        if unfreeze_blocks is not None:
            lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in unfreeze_blocks]
            for block_idx in lora_blocks:
                activate_lora(
                    model.blocks[block_idx],
                    lora_layers,
                    lora_rank,
                    include_attn_key=lora_attn_key,
                    include_attn_proj=lora_attn_proj,
                )
        else:
            activate_lora(model, lora_layers, lora_rank, include_attn_key=lora_attn_key, include_attn_proj=lora_attn_proj)


    def deactivate_lora_before_save(model, lora_cfg):
        model.train(False)  # To merge lora weights
        assert is_merged(model)
        if lora_cfg.unfreeze_blocks is not None:
            lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in lora_cfg.unfreeze_blocks]
            for block_idx in lora_blocks:
                deactivate_lora(model.blocks[block_idx], activate_layers=lora_cfg.layers, delete_separate_proj=False)
        else:
            deactivate_lora(model, activate_layers=lora_cfg.layers, delete_separate_proj=False)  # So that qkv is created


    def reactivate_lora_after_save(model, lora_cfg):
        # Don't need qkv so delete
        if lora_cfg.unfreeze_blocks is not None:
            lora_blocks = [idx for idx in list(range(len(model.blocks))) if idx not in lora_cfg.unfreeze_blocks]
            for block_idx in lora_blocks:
                delete_qkv(model.blocks[block_idx], layers=lora_cfg.layers)
        else:
            delete_qkv(model, layers=lora_cfg.layers)
        model.train(True)
        assert not is_merged(model)
    


