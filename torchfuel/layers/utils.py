import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, imgs):
        batch_size = imgs.size(0)
        return imgs.view(batch_size, -1)


class ReshapeToImg(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, imgs):
        return imgs.view(-1, *self.size)


class PrintLayer(nn.Module):
    def forward(self, inp):
        print(inp.size())
        return inp
