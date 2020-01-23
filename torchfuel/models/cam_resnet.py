import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

from torchfuel.layers.utils import Flatten
from torchfuel.visualisation.visualiser import Visualiser


# Naming is a bit off, but this is just in the process of removing legacy/not useful code
class CAMResnet(Visualiser):
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

        self.gap = nn.Sequential(nn.AvgPool2d(14, 14), Flatten())
        self.fc = nn.Linear(n_filters, n_classes, bias=False)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc(output)
        return output

    def _gen_cam(self, img: torch.Tensor) -> torch.Tensor:
        out = self(img)
        _, pred = torch.max(out, 1)

        activation_maps = self.activations(img).detach()

        b, c, h, w = activation_maps.size()
        activation_maps = activation_maps.view(c, h, w)
        weights = self.fc.weight[pred].detach().view(-1, 1, 1)

        activation_maps = activation_maps * weights
        cam = torch.sum(activation_maps, 0)

        # negative values should be ignored in CAM
        cam = F.relu(cam)

        min_v = torch.min(cam)
        max_v = torch.max(cam)
        range_v = max_v - min_v
        cam = (cam - min_v) / range_v
        cam = cam.cpu().numpy()

        return cam

    def _save_cam_on_image(self, inp_img: str, cam: np.ndarray, out_name: str):
        cam = np.uint8(255 * cam)
        img = cv2.imread(inp_img)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(out_name, result)

    def gen_visualisation(self, img: torch.Tensor, inp_img: str, out_name: str):
        cam = self._gen_cam(img)
        self._save_cam_on_image(inp_img, cam, out_name)
