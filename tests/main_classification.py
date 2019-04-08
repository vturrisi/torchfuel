import os
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from torchfuel.data_loaders.image import ImageDataLoader
from torchfuel.trainers.classification import ClassificationTrainer
from torchfuel.layers.utils import Flatten


class ResnetWithCAM(nn.Module):
    def __init__(self, base_resnet, n_classes):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        last_sublayer = list(self.activations[-1][-1].children())[-1]
        if isinstance(last_sublayer, nn.BatchNorm2d):
            n_filters = last_sublayer.num_features
        elif isinstance(last_sublayer, nn.Conv2d):
            n_filters = last_sublayer.out_channels
        else:
            last_sublayer = list(self.activations[-1][-1].children())[-2]
            n_filters = last_sublayer.num_features

        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc_layer = nn.Linear(n_filters, n_classes, bias=False)

        self.activations.requires_grad = False
        self.gap.requires_grad = False

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output


dl = ImageDataLoader(
    train_data_folder='tests/imgs/train',
    eval_data_folder='tests/imgs/eval',
    pil_transformations=[transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip()],
    batch_size=16,
    imagenet_format=True,
)

train_dataloader, eval_dataloader, n_classes = dl.prepare()

epochs = 1

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

resnet = models.resnet18(pretrained=True).to(device)
model = ResnetWithCAM(resnet, n_classes).to(device)

optimiser = optim.SGD([{'params': model.activations.parameters(), 'lr': 0.005},
                       {'params': model.fc_layer.parameters()}], lr=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20)

trainer = ClassificationTrainer(device, model, optimiser, scheduler)

model_fitted = trainer.fit(epochs, train_dataloader, eval_dataloader)
