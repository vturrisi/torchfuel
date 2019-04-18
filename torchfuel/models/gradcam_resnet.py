import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

from torchfuel.layers.utils import Flatten
from torchfuel.models.cam_model import CAMModel


class GradCAMResnet(CAMModel):
    def __init__(self, base_resnet: ResNet, n_classes: int, resolution: int = 14):
        super().__init__()

        assert resolution in [7, 14, 28, 56, 112]

        self.resolution = resolution
        self._desired_layer_id = 0
        self._desired_layer_output: torch.Tensor = None
        self._desired_layer_grad: torch.Tensor = None

        resnet_children = list(base_resnet.children())

        self.activations = nn.Sequential(*resnet_children[:-1])

        self.flat = Flatten()
        self.fc = nn.Linear(base_resnet.fc.in_features, n_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.activations(imgs)
        output = self.flat(output)
        output = self.fc(output)
        return output

    def _save_output(self, module, input, output):
        # gets the output of the last layer in which its output
        # is the desired resolution
        batch, channels, h, w = output.size()
        if h == self.resolution:
            self._desired_layer_id = id(module)
            self._desired_layer_output = output.detach()

    def _save_grad(self, module, grad_in, grad_out):
        if self._desired_layer_id == id(module):
            self._desired_layer_grad = grad_in[0].detach()

    def _add_hooks(self):
        self._handlers = []
        for module in self.activations.modules():
            self._handlers.append(module.register_forward_hook(self._save_output))
            self._handlers.append(module.register_backward_hook(self._save_grad))

    def _remove_hooks(self):
        for handle in self._handlers:
            handle.remove()
        self._handlers = []

    def get_cam(self, img: torch.Tensor) -> torch.Tensor:
        self._add_hooks()

        out = self(img)
        _, pred = torch.max(out, 1)
        out = out[0, pred]
        out.backward()

        self._remove_hooks()

        grads = self._desired_layer_grad
        activation_maps = self._desired_layer_output

        b, c, h, w = activation_maps.size()
        activation_maps = activation_maps.view(c, h, w)
        weights = torch.mean(grads, (2, 3)).view(-1, 1, 1)

        activation_maps = activation_maps * weights
        cam = torch.sum(activation_maps, 0)
        return cam
