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

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20)

trainer = ClassificationTrainer(device, model, optimiser, scheduler)

epochs = 1
model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
error = trainer.state.train_loss

epochs = 2
model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
new_error = trainer.state.train_loss

assert new_error < error
