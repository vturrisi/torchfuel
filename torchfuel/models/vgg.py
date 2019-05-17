from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16


class SlicedVGG(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg16(pretrained=True).features

        cuts = [0]
        for i, model in enumerate(vgg.children()):
            if isinstance(model, torch.nn.ReLU):
                cuts.append(i + 1)

        self.vgg = nn.ModuleList()
        for i, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            group = nn.Sequential(*vgg[start:end])
            self.vgg.add_module(str(i), group)

    def forward(self, inp):
        vgg_outputs = namedtuple("VggOutputs", [f'relu{i}' for i in range(len(self.vgg))])
        temp = []
        h = inp
        for module in self.vgg:
            h = module(h)
            temp.append(h)
        out = vgg_outputs(*temp)
        return out
