import pytest
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from torchfuel.data_loaders.image import (ImageDataLoader,
                                          ImageToImageDataLoader)
from torchfuel.transforms.noise import DropPixelNoiser, GaussianNoiser


def test_imagedl():
    dl = ImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader, eval_dataloader, n_classes = dl.prepare()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)
    assert isinstance(n_classes, int)

def test_image2imagedl():
    dl = ImageToImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        tensor_transformations=[DropPixelNoiser(noise_chance=0.1),
                                GaussianNoiser(noise_amount=0.1)],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader, eval_dataloader = dl.prepare()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)


def test_loader_error():
    with pytest.raises(Exception):
        dl = ImageToImageDataLoader(
            train_data_folder='tests/imgs/train',
            eval_data_folder='tests/imgs/eval',
            batch_size=16,
        )

def test_loader_normalize_not_imagenet():
    dl = ImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        batch_size=16,
        size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1],
    )

    train_dataloader, eval_dataloader, n_classes = dl.prepare()
    it = iter(train_dataloader)
    assert next(it)


def test_loader_apply_to_eval():
    dl = ImageDataLoader(
        train_data_folder='tests/imgs/train',
        eval_data_folder='tests/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        tensor_transformations=[DropPixelNoiser(noise_chance=0.1),
                                GaussianNoiser(noise_amount=0.1)],
        apply_pil_transforms_to_eval=True,
        apply_tensor_transforms_to_eval=True,
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader, eval_dataloader, n_classes = dl.prepare()
    it = iter(train_dataloader)
    assert next(it)

test_loader_apply_to_eval()
