import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms

from torchfuel.data_loaders.image import ImageDataLoader, ImageToImageDataLoader
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

    train_dataloader, eval_dataloader, n_classes = dl.prepare()

    device = torch.device('cpu')

    resnet = models.resnet18(pretrained=True)
    model = CAMResnet(resnet, n_classes).to(device)

    it = iter(train_dataloader)
    X, y = next(it)
    assert isinstance(model(X), torch.Tensor)
