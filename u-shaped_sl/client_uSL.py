
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

class ClientUShapedSL:
    def __init__(
        self,
        head: nn.Module,
        tail: nn.Module,
        DataLoader: torch.utils.data.DataLoader
    ):
        """
        Args:
            head (nn.Module): head module, e.g. a first layers on the client
            tail (nn.Module): tail module, e.g. a final layers on the client
        """
        self.head: nn.Module = head
        self.tail: nn.Module = tail
        self.dataLoader= DataLoader
        self.dataiterator = iter(self.dataLoader)

    def forwardHead(self):
        try:
            batch = next(self.dataiterator)
        except StopIteration:
            self.dataiterator = iter(self.dataLoader)
            batch = next(self.dataiterator)
        x = self.head(batch)
        return x
        

