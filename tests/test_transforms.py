from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from torchfuel.data_loaders.image import ImageDataLoader, ImageToImageDataLoader


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
        batch_size=16,
        imagenet_format=True,
    )

    train_dataloader, eval_dataloader = dl.prepare()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)
