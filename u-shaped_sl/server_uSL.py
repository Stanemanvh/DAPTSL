import torch.nn as nn


class ServerUShapedSL:
    def __init__(self, backbone: nn.Module):
        """
        Args:
            backbone (nn.Module): server-side backbone module (e.g., DinoBackbone)
        """
        self.backbone = backbone

    def forwardBackbone(self, client_embeddings):
        return self.backbone(client_embeddings)
