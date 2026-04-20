# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# --------------------------------------------------------

import math
import logging
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from dinov2.layers import PatchEmbed

logger = logging.getLogger("dinov2")


class DinoBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
    ):
        super().__init__()
        self.head = backbone


    def forward(self, x, masks=None):
        global_crops = x["collated_global_crops"].cuda(non_blocking=True)
        local_crops = x["collated_local_crops"].cuda(non_blocking=True)
        masks = x["collated_masks"].cuda(non_blocking=True)

        global_unmasked = self.head(global_crops, masks=None)
        global_masked = self.head(global_crops, masks=masks)
        local_unmasked = self.head(local_crops, masks=None)

        return global_unmasked, global_masked, local_unmasked, masks
    


