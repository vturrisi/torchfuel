import os
import shutil
import sys
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

from torchfuel.data_loaders.image import ImageToImageDataLoader
from torchfuel.layers.utils import Flatten, ReshapeToImg
from torchfuel.trainers.mse import MSETrainer
from torchfuel.transforms.noise import DropPixelNoiser, GaussianNoiser
from torchvision import datasets, models, transforms


def test():
    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = nn.Sequential(
                # 56x56
                Flatten(),
                nn.Linear(56 * 56 * 3, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(50, 56 * 56 * 3), nn.Sigmoid(), ReshapeToImg(3, 56, 56),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    device = torch.device("cpu")

    dl = ImageToImageDataLoader(
        train_data_folder="test/imgs/train",
        eval_data_folder="test/imgs/eval",
        tensor_transformations=[],
        size=56,
        imagenet_format=False,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl

    model = AutoEncoder().to(device)

    optimiser = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", patience=50)

    epochs = 10
    trainer = MSETrainer(
        device,
        model,
        optimiser,
        scheduler,
        checkpoint_model=True,
        model_name="test/autoencoder.pt",
    )

    trainer.fit(epochs, train_dataloader, eval_dataloader)

    trainer.epochs = 20
    # will need to load model
    trainer.fit(epochs, train_dataloader, eval_dataloader)

    os.remove("test/autoencoder.pt")


if __name__ == "__main__":
    test()
