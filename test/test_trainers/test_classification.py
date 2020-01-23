import os
import sys
import time
from collections import namedtuple
from contextlib import suppress

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

from torchfuel.data_loaders.image import ImageDataLoader
from torchfuel.layers.utils import Flatten
from torchfuel.models.cam_resnet import CAMResnet
from torchfuel.trainers.classification import ClassificationTrainer
from torchvision import datasets, models, transforms


def test():
    dl = ImageDataLoader(
        train_data_folder="test/imgs/train",
        eval_data_folder="test/imgs/eval",
        test_data_folder="test/imgs/eval",
        pil_transformations=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    test_dataloader = dl.test_dl
    n_classes = dl.n_classes

    device = torch.device("cpu")

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    optimiser = optim.Adam(
        [
            {"params": model.activations.parameters(), "lr": 0.005},
            {"params": model.fc.parameters()},
        ],
        lr=0.01,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=1)

    trainer = ClassificationTrainer(
        device,
        model,
        optimiser,
        scheduler,
        checkpoint_model=True,
        model_name="test/model.pt",
        compute_confusion_matrix=True,
        n_classes=n_classes,
    )

    with suppress(FileNotFoundError):
        os.remove("test/model.pt")
    epochs = 2
    trainer.fit(epochs, train_dataloader, eval_dataloader)
    error1 = trainer.state.train_loss

    epochs = 3
    trainer.fit(epochs, train_dataloader, eval_dataloader)
    error2 = trainer.state.train_loss
    cm = trainer.state.train_cm

    assert isinstance(cm, torch.Tensor)

    assert error1 > error2

    trainer.test(test_dataloader, load=False)
    trainer.test(test_dataloader, load=True)

    assert trainer.state.test

    print(trainer.state.test)

    os.remove("test/model.pt")


if __name__ == "__main__":
    test()
