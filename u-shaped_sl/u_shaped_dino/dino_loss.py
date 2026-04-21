# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn

from dinov2.loss import DINOLoss, KoLeoLoss


logger = logging.getLogger("dinov2")


class DinoLoss(nn.Module):
    """
    Loss module that consumes DinoBackbone outputs.

    Expected input format:
      backbone_output["loss_inputs"] with keys:
            - teacher_dino_softmaxed_centered_list
            - masked_teacher_ibot_softmaxed_centered
            - student_local_cls_tokens_after_head
            - student_global_cls_tokens_after_head
            - student_global_masked_patch_tokens_after_head
            - student_global_cls_tokens
            - student_masks
            - n_masked_patches
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        self.dino_loss_weight = cfg.dino.loss_weight
        self.ibot_loss_weight = cfg.ibot.loss_weight
        self.ibot_student_temp = getattr(cfg.ibot, "student_temp", 0.1)
        self.koleo_loss_weight = cfg.dino.koleo_loss_weight
        self.skip_koleo_inf = getattr(cfg.dino, "skip_koleo_inf", False)
        self.koleo_eps = getattr(cfg.dino, "koleo_eps", 1e-8)

        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.dino_loss = DINOLoss(self.dino_out_dim)
        self.koleo_loss = KoLeoLoss(dist_eps=self.koleo_eps) if self.do_koleo else None

    def _compute_dino_local_loss(self, student_local_logits, teacher_dino_targets, n_local_crops, n_global_crops):
        if n_local_crops <= 0:
            return None

        n_local_terms = max(n_local_crops * n_global_crops, 1)
        n_global_terms = (n_global_crops - 1) * n_global_crops

        return self.dino_loss(
            student_output_list=student_local_logits.chunk(n_local_crops),
            teacher_out_softmaxed_centered_list=teacher_dino_targets,
        ) / (n_local_terms + n_global_terms)

    def _compute_dino_global_loss(self, student_global_logits, teacher_dino_targets, n_local_crops, n_global_crops):
        n_local_terms = max(n_local_crops * n_global_crops, 1)
        n_global_terms = (n_global_crops - 1) * n_global_crops

        return (
            self.dino_loss(
                student_output_list=[student_global_logits],
                teacher_out_softmaxed_centered_list=[teacher_dino_targets.flatten(0, 1)],
            )
            * 2
            / (n_local_terms + n_global_terms)
        )

    def _compute_ibot_loss(self, student_patch_logits, teacher_ibot_targets, student_masks, n_masked_patches):
        if not self.do_ibot:
            return None

        ibot_loss = torch.sum(
            teacher_ibot_targets * torch.log_softmax(student_patch_logits / self.ibot_student_temp, dim=-1),
            dim=-1,
        )
        masks_weight = (
            (1 / student_masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(student_masks)[student_masks.bool()]
        )
        ibot_loss = ibot_loss[:n_masked_patches] * masks_weight
        return -ibot_loss.sum() / student_masks.shape[0]

    def _compute_koleo_loss(self, student_global_cls_tokens):
        if not self.do_koleo:
            return None

        koleo_loss = self.koleo_loss_weight * sum(
            self.koleo_loss(p) for p in student_global_cls_tokens.chunk(2)
        )
        if self.skip_koleo_inf and torch.isinf(koleo_loss):
            koleo_loss = torch.zeros_like(koleo_loss)
        return koleo_loss

    def forward(self, backbone_output, teacher_temp=None):
        loss_inputs = backbone_output["loss_inputs"]

        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number

        teacher_dino_targets = loss_inputs["teacher_dino_softmaxed_centered_list"]
        teacher_ibot_targets = loss_inputs["masked_teacher_ibot_softmaxed_centered"]

        student_local_logits = loss_inputs["student_local_cls_tokens_after_head"]
        student_global_logits = loss_inputs["student_global_cls_tokens_after_head"]
        student_patch_logits = loss_inputs["student_global_masked_patch_tokens_after_head"]

        loss_dict = {}
        total_loss = student_global_logits.new_zeros(())

        if self.do_dino:
            dino_local_loss = self._compute_dino_local_loss(
                student_local_logits=student_local_logits,
                teacher_dino_targets=teacher_dino_targets,
                n_local_crops=n_local_crops,
                n_global_crops=n_global_crops,
            )
            if dino_local_loss is not None:
                loss_dict["dino_local_crops_loss"] = dino_local_loss
                total_loss = total_loss + self.dino_loss_weight * dino_local_loss

            dino_global_loss = self._compute_dino_global_loss(
                student_global_logits=student_global_logits,
                teacher_dino_targets=teacher_dino_targets,
                n_local_crops=n_local_crops,
                n_global_crops=n_global_crops,
            )
            loss_dict["dino_global_crops_loss"] = dino_global_loss
            total_loss = total_loss + self.dino_loss_weight * dino_global_loss

            koleo_loss = self._compute_koleo_loss(loss_inputs["student_global_cls_tokens"])
            if koleo_loss is not None:
                loss_dict["koleo_loss"] = koleo_loss / 2
                total_loss = total_loss + koleo_loss

        if self.do_ibot:
            ibot_loss = self._compute_ibot_loss(
                student_patch_logits=student_patch_logits,
                teacher_ibot_targets=teacher_ibot_targets,
                student_masks=loss_inputs["student_masks"],
                n_masked_patches=loss_inputs["n_masked_patches"],
            )
            loss_dict["ibot_loss"] = ibot_loss
            total_loss = total_loss + self.ibot_loss_weight * ibot_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def backward(self, loss_dict, accum_iter=1, scaler=None):
        total_loss = loss_dict["total_loss"] / accum_iter
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return total_loss
