import pytest
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from torchfuel.data_loaders.image import (ImageDataLoader,
                                          ImageToImageDataLoader)
from torchfuel.transforms.noise import DropPixelNoiser, GaussianNoiser


def test_imagedl():
    dl = ImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    n_classes = dl.n_classes

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)
    assert isinstance(n_classes, int)


def test_image2imagedl():
    dl = ImageToImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        tensor_transformations=[DropPixelNoiser(noise_chance=0.1),
                                GaussianNoiser(noise_amount=0.1)],
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)


def test_loader_error():
    with pytest.raises(Exception):
        dl = ImageToImageDataLoader(
            train_data_folder='test/imgs/train',
            eval_data_folder='test/imgs/eval',
            batch_size=16,
        )


def test_loader_normalize_not_imagenet():
    dl = ImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        batch_size=16,
        size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1],
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    n_classes = dl.n_classes
    it = iter(train_dataloader)
    assert next(it)


def test_loader_force_prepare_error():
    dl = ImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        batch_size=16,
        size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1],
    )

    with pytest.raises(Exception):
        dl.size = None
        dl.mean = None
        dl.std = None
        dl.prepare()


def test_loader_apply_to_eval():
    dl = ImageDataLoader(
        train_data_folder='test/imgs/train',
        eval_data_folder='test/imgs/eval',
        test_data_folder='test/imgs/eval',
        pil_transformations=[transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()],
        tensor_transformations=[DropPixelNoiser(noise_chance=0.1),
                                GaussianNoiser(noise_amount=0.1)],

        pil_transformations_eval=[transforms.RandomHorizontalFlip()],
        tensor_transformations_eval=[DropPixelNoiser(noise_chance=0.1)],

        pil_transformations_test=[transforms.RandomHorizontalFlip()],
        tensor_transformations_test=[DropPixelNoiser(noise_chance=0.1)],

        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader = dl.train_dl
    eval_dataloader = dl.eval_dl
    n_classes = dl.n_classes
    it = iter(train_dataloader)
    assert next(it)


if __name__ == '__main__':
    test_imagedl()
    test_image2imagedl()
    test_loader_error()
    test_loader_normalize_not_imagenet()
    test_loader_force_prepare_error()
    test_loader_apply_to_eval()
