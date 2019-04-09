import torch.nn as nn

from torchfuel.layers.utils import Flatten


class CAMResnet(nn.Module):
    def __init__(self, base_resnet, n_classes):
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
        self.fc_layer = nn.Linear(n_filters, n_classes, bias=False)

        self.activations.requires_grad = False
        self.gap.requires_grad = False

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output
