import os
import shutil

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms

from torchfuel.data_loaders.image import (ImageDataLoader,
                                          ImageToImageDataLoader)
from torchfuel.models.cam_resnet import CAMResnet


def test_camresnet():
    dl = ImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    n_classes = dl.n_classes

    device = torch.device('cpu')

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    it = iter(train_dataloader)
    X, y = next(it)
    assert isinstance(model(X), torch.Tensor)

    resnet = models.resnet50(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    it = iter(train_dataloader)
    X, y = next(it)
    assert isinstance(model(X), torch.Tensor)

    cam_folder = 'tests/cams'
    model.gen_cams(device, 'tests/imgs/train', cam_folder,
                   minmax=True, imagenet=True)

    cam_per_label = os.listdir(cam_folder)
    assert cam_per_label

    for label in cam_per_label:
        assert os.listdir(os.path.join(cam_folder, label))

    shutil.rmtree(cam_folder, ignore_errors=True)


if __name__ == '__main__':
    test_camresnet()
