import os
import time
from collections import namedtuple
from contextlib import suppress

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
        test_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    test_dataloader = dl.test_dl
    n_classes = dl.n_classes

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    optimiser = optim.SGD([{'params': model.activations.parameters(), 'lr': 0.005},
                           {'params': model.fc_layer.parameters()}], lr=0.01, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=1)

    errors = []
    trainer = ClassificationTrainer(
        device,
        model,
        optimiser,
        scheduler,
        checkpoint_model=True,
        model_name='tests/model.pt',
        compute_confusion_matrix=True,
        n_classes=n_classes
    )

    with suppress(FileNotFoundError):
        os.remove('tests/model.pt')
    epochs = 2
    model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
    error1 = trainer.state.train_loss

    epochs = 3
    model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
    error2 = trainer.state.train_loss
    cm = trainer.state.train_cm

    assert isinstance(cm, torch.Tensor)

    assert error1 > error2

    trainer.test(test_dataloader, load=False)
    trainer.test(test_dataloader, load=True)

    assert trainer.state.test

    print(trainer.state.test)

    os.remove('tests/model.pt')


if __name__ == '__main__':
    test()
