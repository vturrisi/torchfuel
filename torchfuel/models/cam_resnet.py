import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

from torchfuel.layers.utils import Flatten
from torchfuel.models.cam_model import CAMModel


class CAMResnet(CAMModel):
    def __init__(self, base_resnet: ResNet, n_classes: int):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        last_sublayer = list(self.activations[-1][-1].children())[-1]
        if isinstance(last_sublayer, nn.BatchNorm2d):
            n_filters = last_sublayer.num_features
        elif isinstance(last_sublayer, nn.Conv2d):
            n_filters = last_sublayer.out_channels
        else:
            last_sublayer = list(self.activations[-1][-1].children())[-2]
            n_filters = last_sublayer.num_features

        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc = nn.Linear(n_filters, n_classes, bias=False)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc(output)
        return output

    def get_cam(self, img: torch.Tensor) -> torch.Tensor:
        out = self(img)
        _, pred = torch.max(out, 1)

        activation_maps = self.activations(img).detach()

        b, c, h, w = activation_maps.size()
        activation_maps = activation_maps.view(c, h, w)
        weights = self.fc.weight[pred].detach().view(-1, 1, 1)

        activation_maps = activation_maps * weights
        cam = torch.sum(activation_maps, 0)
        return cam
