import os
import shutil
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from torchfuel.data_loaders.image import ImageToImageDataLoader
from torchfuel.layers.utils import Flatten, ReshapeToImg
from torchfuel.trainers.perceptual_loss import PerceptualLossTrainer
from torchfuel.transforms.noise import DropPixelNoiser, GaussianNoiser


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
                nn.Linear(50, 56 * 56 * 3),
                nn.Sigmoid(),
                ReshapeToImg(3, 56, 56),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    device = torch.device('cpu')

    dl = ImageToImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        tensor_transformations=[],
        size=56,
        imagenet_format=False
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl

    model = AutoEncoder().to(device)

    optimiser = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=50)

    epochs = 30
    trainer = PerceptualLossTrainer(
        device,
        model,
        optimiser,
        scheduler,
        checkpoint_model=True,
        model_name='test/autoencoder.pt',
        use_avg_loss=True,
    )

    trainer.fit(epochs, train_dataloader, eval_dataloader)

    os.remove('test/autoencoder.pt')


if __name__ == '__main__':
    test()
