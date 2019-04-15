import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

from torchfuel.layers.utils import Flatten
from torchfuel.models.cam_model import CAMModel
import numpy as np

class GradCAMResnet(CAMModel):
    def __init__(self, base_resnet: ResNet, n_classes: int):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-2])
        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc = nn.Linear(base_resnet.fc.in_features, n_classes)

        def save_grad(module, grad_in, grad_out):
            self.grads = {'grad_in': grad_in, 'grad_out': grad_out}

        # list(self.activations.modules())[-3].register_backward_hook(save_grad)
        self.activations.register_backward_hook(save_grad)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc(output)
        return output

    def get_cam(self, img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        out = self(img)
        out = out[0, label]
        out.backward()
        grads = self.grads['grad_in'][-1]
        activation_maps = self.activations(img).detach()
        b, c, h, w = activation_maps.size()
        activation_maps = activation_maps.view(c, h, w)
        weights = torch.mean(grads, (2, 3)).view(-1, 1, 1)
        weights = torch.abs(weights)
        activation_maps = activation_maps * weights
        cam = torch.sum(activation_maps, 0)
        *_, i, j = cam.size()
        return cam
