# torchfuel
[![Build Status](https://travis-ci.org/vturrisi/torchfuel.svg?branch=master)](https://travis-ci.org/vturrisi/torchfuel)
[![codecov](https://codecov.io/gh/vturrisi/torchfuel/branch/master/graph/badge.svg)](https://codecov.io/gh/vturrisi/torchfuel)

Build on top of pytorch to fuel productivity.

# Features

- Generic Trainer
- Classification Trainer (with cross-entropy loss)
- MSE Trainer
- Additional utility layers
- Better dataloaders (currently only for image datasets)

# Classification Example

```python
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
from torchfuel.transforms.noise import DropPixelNoiser


dl = ImageDataLoader(
    train_data_folder='imgs/train',
    eval_data_folder='imgs/eval',
    pil_transformations=[transforms.RandomHorizontalFlip()]
    tensor_transformations=[DropPixelNoiser()],
    batch_size=64,
    imagenet_format=True,
)

train_dataloader, eval_dataloader, n_classes = dl.prepare()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Model(...).to(device)

optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20)

trainer = ClassificationTrainer(device, model, optimiser, scheduler)

fitted_model = trainer.fit(epochs, train_dataloader, eval_dataloader)

```

# How to install
Clone repository and run:
```bash
pip install .
```

Optionally (not up to date):
```bash
pip install torchfuel
```

