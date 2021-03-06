import os
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

from torchfuel.data_loaders.image import ImageDataLoader, ImageToImageDataLoader
from torchfuel.models.cam_resnet import CAMResnet
from torchfuel.layers.utils import PrintLayer, ReshapeToImg, Flatten


def test_util_layers():
    dl = ImageDataLoader(
        train_data_folder="test/imgs/train",
        eval_data_folder="test/imgs/eval",
        pil_transformations=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ],
        batch_size=16,
        imagenet_format=True,
    )

    class UtilLayerTester(nn.Module):
        def __init__(self):
            super().__init__()
            self.pl = PrintLayer()
            self.flat = Flatten()
            self.r2i = ReshapeToImg()

        def forward(self, inp):
            inp = self.pl(inp)
            inp = self.flat(inp)
            inp = self.r2i(inp)
            return inp

    train_dataloader = dl.train_dl

    device = torch.device("cpu")
    model = UtilLayerTester().to(device)

    it = iter(train_dataloader)
    X, y = next(it)
    assert isinstance(model(X), torch.Tensor)


if __name__ == "__main__":
    test_util_layers()
