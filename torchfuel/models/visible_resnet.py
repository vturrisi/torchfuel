import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

from torchfuel.layers.utils import Flatten


class VisibleResnet(nn.Module):
    def __init__(self, base_resnet: ResNet, n_classes: int):
        super().__init__()

        resnet_children = list(base_resnet.children())

        self.activations = nn.Sequential(*resnet_children[:-1])
        self.flat = Flatten()
        self.fc = nn.Linear(base_resnet.fc.in_features, n_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.flat(self.activations(imgs))
        output = self.fc(output)
        return output
