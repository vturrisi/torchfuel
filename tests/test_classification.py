import os
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from torchfuel.data_loaders.image import ImageDataLoader
from torchfuel.layers.utils import Flatten
from torchfuel.models.cam_resnet import CAMResnet
from torchfuel.trainers.classification import ClassificationTrainer


def test():
    dl = ImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader, eval_dataloader, n_classes = dl.prepare()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    optimiser = optim.SGD([{'params': model.activations.parameters(), 'lr': 0.005},
                           {'params': model.fc_layer.parameters()}], lr=0.01, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=1)

    epochs = 1
    errors = []
    for i in range(2):
        trainer = ClassificationTrainer(
            device,
            model,
            optimiser,
            scheduler,
            checkpoint_model=False,
            compute_confusion_matrix=True,
            n_classes=n_classes
        )

        model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
        error = trainer.state.train_loss
        cm = trainer.state.train_cm
        assert isinstance(cm, torch.Tensor)
        errors.append(error)

    assert errors[0] > errors[1]
