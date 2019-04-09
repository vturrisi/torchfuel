import os
import time
from collections import namedtuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from torchfuel.data_loaders.image import ImageToImageDataLoader
from torchfuel.layers.utils import Flatten, ReshapeToImg
from torchfuel.trainers.const import AFTER_EPOCH
from torchfuel.trainers.mse import MSETrainer
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

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    dl = ImageToImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        tensor_transformations=[],
        size=56,
        imagenet_format=False
    )

    train_dataloader, eval_dataloader = dl.prepare()

    model = AutoEncoder().to(device)

    optimiser = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=50)

    epochs = 1
    trainer = MSETrainer(
        device,
        model,
        optimiser,
        scheduler,
        checkpoint_model=False,
    )

    @trainer.execute_on(AFTER_EPOCH)
    def dummy_function(trainer):
        pass

    with pytest.raises(Exception):
        @trainer.execute_on(1000)
        def dummy_function(trainer):
            pass

    with pytest.raises(Exception):
        trainer._run_hooks(1000)

    with pytest.raises(Exception):
        trainer.add_hook(dummy_function, 1000)
