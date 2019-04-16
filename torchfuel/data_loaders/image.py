from typing import Iterable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ImageDataLoader:
    def __init__(self,
                 train_data_folder: str,
                 eval_data_folder: str,
                 test_data_folder: Optional[str] = None,
                 pil_transformations: Optional[Iterable] = None,
                 tensor_transformations: Optional[Iterable] = None,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 imagenet_format=False,
                 size: Union[int, Iterable] = None,
                 mean: Optional[Iterable] = None,
                 std: Optional[Iterable] = None,
                 pil_transformations_eval: Optional[Iterable] = None,
                 tensor_transformations_eval: Optional[Iterable] = None,
                 pil_transformations_test: Optional[Iterable] = None,
                 tensor_transformations_test: Optional[Iterable] = None,
                 dataset_class: Dataset = datasets.ImageFolder):

        if pil_transformations is None:
            pil_transformations = []

        if tensor_transformations is None:
            tensor_transformations = []

        self.train_data_folder = train_data_folder
        self.eval_data_folder = eval_data_folder
        self.test_data_folder = test_data_folder
        self.pil_transformations = pil_transformations
        self.tensor_transformations = tensor_transformations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.imagenet_format = imagenet_format
        self.size = size
        self.mean = mean
        self.std = std

        self.pil_transformations_eval = pil_transformations_eval
        self.tensor_transformations_eval = tensor_transformations_eval

        self.pil_transformations_test = pil_transformations_test
        self.tensor_transformations_test = tensor_transformations_test

        self.dataset_class = dataset_class

        if self.imagenet_format:
            assert all(v is None for v in [self.size, self.mean, self.std])
        else:
            assert self.size is not None, 'imagenet_format should be True or size must be specified'

        # prepare data
        self.prepare()

    def prepare(self):
        if self.imagenet_format:
            size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalise = True
        elif all(v is not None for v in [self.size, self.mean, self.std]):
            size = self.size
            mean = self.mean
            std = self.std
            normalise = True
        elif self.size is not None:
            size = self.size
            normalise = False
        else:
            raise Exception('imagenet_format should be True or size must be specified')

        # train transforms
        train_t = [
            transforms.Resize(size),
            *self.pil_transformations,
            transforms.ToTensor(),
        ]
        if normalise:
            train_t.append(transforms.Normalize(mean, std))
        train_t.extend(self.tensor_transformations)

        train_transforms = transforms.Compose(train_t)

        if self.pil_transformations_eval:
            eval_t = [
                transforms.Resize(size),
                *self.pil_transformations_eval,
                transforms.ToTensor(),
            ]
        else:
            eval_t = [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
        if normalise:
            eval_t.append(transforms.Normalize(mean, std))

        if self.tensor_transformations_eval:
            eval_t.extend(self.tensor_transformations_eval)

        eval_transforms = transforms.Compose(eval_t)

        if self.test_data_folder:
            if self.pil_transformations_test:
                test_t = [
                    transforms.Resize(size),
                    *self.pil_transformations_test,
                    transforms.ToTensor(),
                ]
            else:
                test_t = [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            if normalise:
                test_t.append(transforms.Normalize(mean, std))

            if self.tensor_transformations_test:
                test_t.extend(self.tensor_transformations_test)

            test_transforms = transforms.Compose(test_t)

        train_data = self.dataset_class(self.train_data_folder, train_transforms)
        train_dataloader = DataLoader(train_data, self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers)

        eval_data = self.dataset_class(self.eval_data_folder, eval_transforms)
        eval_dataloader = DataLoader(eval_data, self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)

        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader

        class_names = train_data.classes
        n_classes = len(class_names)
        self.n_classes = n_classes

        if self.test_data_folder:
            test_data = self.dataset_class(self.test_data_folder, test_transforms)
            test_dataloader = DataLoader(test_data, self.batch_size,
                                         shuffle=False, num_workers=self.num_workers)
            self.test_dl = test_dataloader


class ImageToImageReader(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, y = super().__getitem__(index)
        return img, img


class ImageToImageDataLoader(ImageDataLoader):
    def __init__(self,
                 train_data_folder: str,
                 eval_data_folder: str,
                 test_data_folder: str = None,
                 pil_transformations: Optional[list] = None,
                 tensor_transformations: Optional[list] = None,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 imagenet_format=False,
                 size: Union[int, Iterable] = None,
                 mean: Optional[Iterable] = None,
                 std: Optional[Iterable] = None,
                 pil_transformations_eval: Optional[Iterable] = None,
                 tensor_transformations_eval: Optional[Iterable] = None,
                 pil_transformations_test: Optional[Iterable] = None,
                 tensor_transformations_test: Optional[Iterable] = None,
                 dataset_class: Dataset = datasets.ImageFolder
                 ):

        super().__init__(
            train_data_folder,
            eval_data_folder,
            test_data_folder,
            pil_transformations,
            tensor_transformations,
            batch_size,
            shuffle,
            num_workers,
            imagenet_format,
            size,
            mean,
            std,
            pil_transformations_eval,
            tensor_transformations_eval,
            pil_transformations_test,
            tensor_transformations_test,
            dataset_class=ImageToImageReader
        )

    def prepare(self):
        super().prepare()
        del self.n_classes

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        return (path, *original_tuple)
