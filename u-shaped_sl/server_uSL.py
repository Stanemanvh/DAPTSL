import torch.nn as nn


class ServerUShapedSL:
    def __init__(self, backbone: nn.Module, dino_loss: nn.Module = None):
        """
        Args:
            backbone (nn.Module): server-side backbone module (e.g., DinoBackbone)
            dino_loss (nn.Module, optional): loss module consuming DinoBackbone outputs
        """
        self.backbone = backbone
        self.dino_loss = dino_loss

    def forwardBackbone(self, client_embeddings, teacher_temp=None):
        return self.backbone.forward_features(client_embeddings, teacher_temp=teacher_temp)

    def computeLoss(self, backbone_output, teacher_temp):
        if self.dino_loss is None:
            raise RuntimeError("No dino_loss module was injected into ServerUShapedSL.")
        return self.dino_loss(backbone_output=backbone_output, teacher_temp=teacher_temp)

    def forwardBackboneAndComputeLoss(self, client_embeddings, teacher_temp):
        backbone_output = self.forwardBackbone(client_embeddings, teacher_temp=teacher_temp)
        loss_dict = self.computeLoss(backbone_output=backbone_output, teacher_temp=teacher_temp)
        return backbone_output, loss_dict

    def backwardLoss(self, loss_dict, accum_iter=1, scaler=None):
        if self.dino_loss is None:
            raise RuntimeError("No dino_loss module was injected into ServerUShapedSL.")
        return self.dino_loss.backward(loss_dict=loss_dict, accum_iter=accum_iter, scaler=scaler)
