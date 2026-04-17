
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

class UShapedSL:
    def __init__(
        self,
        nr_of_clients: int,
        head: nn.Module,
        backbone: nn.Module,
        tail: nn.Module
    ):
        """
        Args:
            nr_of_clients (int): amount of clients participating in the training
            head (nn.Module): head module, e.g. a first layers on the client
            backbone (nn.Module): backbone module, e.g. a backbone on the server
            tail (nn.Module): tail module, e.g. a final layers on the client
        """
        self.head: nn.Module = head
        self.backbone: nn.Module = backbone
        self.tail: nn.Module = tail
        self.nr_of_clients: int = nr_of_clients

    def forwardHeads(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
        
    def forwardBackbone(self, *args, **kwargs):
        return self.forward_features(*args, **kwargs)
    
    def forwardTail(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return self.tail(ret["x_tokens_list"])
    
    def evaluateBackboneLoss(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_clstoken"], ret["x_tokens_list"]
    
    def evaluateTailLoss(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_tokens_list"]

    def backpropagateBackbone(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_clstoken"], ret["x_tokens_list"]
    
    def backpropagateTail(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_tokens_list"]
    
    def updateBackbone(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_clstoken"], ret["x_tokens_list"]
    
    def updateTail(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_tokens_list"]
    
    def clone_clients(self):
        """
        Clone the head and tail modules for each client, so that they can be trained separately.
        """
        self.head = nn.ModuleList([self.head for _ in range(self.nr_of_clients)])
        self.tail = nn.ModuleList([self.tail for _ in range(self.nr_of_clients)])
    
    def train(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_clstoken"], ret["x_tokens_list"]

