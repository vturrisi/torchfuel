import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, imgs):
        return imgs.view(-1, 56 * 56 * 3)


class ReshapeToImg(nn.Module):
    def forward(self, imgs):
        return imgs.view(-1, 3, 56, 56)
