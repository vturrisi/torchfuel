import os
import shutil

import pytest
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

    it = iter(train_dataloader)
    X, y = next(it)

    resnet = models.resnet50(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    resnet = models.resnet101(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    assert isinstance(model(X), torch.Tensor)

    img_folder = 'tests/imgs/train'
    cam_folder = 'tests/cams'

    shutil.rmtree(cam_folder, ignore_errors=True)

    model.gen_cams(device, img_folder, cam_folder, imagenet=True)

    cams = os.listdir(cam_folder)
    imgs = []
    for subfolder in os.listdir(img_folder):
        imgs.extend(os.listdir(os.path.join(img_folder, subfolder)))
    assert cams

    assert len(cams) == len(imgs)

    shutil.rmtree(cam_folder, ignore_errors=True)

    model.gen_cams(device, img_folder, cam_folder,
                   normalise_abs=True, size=224)

    cams = os.listdir(cam_folder)
    imgs = []
    for subfolder in os.listdir(img_folder):
        imgs.extend(os.listdir(os.path.join(img_folder, subfolder)))
    assert cams

    assert len(cams) == len(imgs)

    shutil.rmtree(cam_folder, ignore_errors=True)

    with pytest.raises(Exception):
        model.gen_cams(device, img_folder, cam_folder, minmax=True)


if __name__ == '__main__':
    test_camresnet()
