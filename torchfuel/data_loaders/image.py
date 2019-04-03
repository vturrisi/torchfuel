import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


class ImageDataLoader:
    def __init__(self,
                 train_data_folder: str, eval_data_folder: str,
                 pil_transformations: list = None,
                 tensor_transformations: list = None,
                 batch_size=64,
                 shuffle=True,
                 num_workers=4,
                 imagenet_format=True,
                 size=None, mean: list = None, std: list = None,
                 apply_transforms_to_eval: bool = False,
                 dataset_class: Dataset = datasets.ImageFolder):

        if pil_transformations is None:
            pil_transformations = []

        if tensor_transformations is None:
            tensor_transformations = []

        self.train_data_folder = train_data_folder
        self.eval_data_folder = eval_data_folder
        self.pil_transformations = pil_transformations
        self.tensor_transformations = tensor_transformations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.imagenet_format = imagenet_format
        self.size = size
        self.mean = mean
        self.std = std
        self.apply_transforms_to_eval = apply_transforms_to_eval

        self.dataset_class = dataset_class

    def prepare(self):
        if self.imagenet_format:
            assert all(v is None for v in [self.size, self.mean, self.std])

            size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalise = True
        elif all(v is not None for v in [self.size, self.mean, self.std]):
            size = self.size
            mean = self.mean
            std = self.std
            normalise = False
        elif self.size is not None:
            size = self.size
            normalise = False
        else:
            raise Exception('imagenet_format should be True or size must be specified')

        t = [
            transforms.Resize(size),
            *self.pil_transformations,
            transforms.ToTensor(),
        ]
        if normalise:
            t.append(transforms.Normalize(mean, std))
        t.extend(self.tensor_transformations)

        train_transforms = transforms.Compose(t)

        if self.apply_transforms_to_eval:
            eval_transforms = train_transforms
        else:
            t = [
                transforms.Resize(size),
                transforms.ToTensor(),
            ]
            if normalise:
                t.append(transforms.Normalize(mean, std))
            eval_transforms = transforms.Compose(t)

        data_transforms = {
            'train': train_transforms,
            'eval': eval_transforms,
        }

        train_data = self.dataset_class(self.train_data_folder, data_transforms['train'])
        train_dataloader = DataLoader(train_data, self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers)

        eval_data = self.dataset_class(self.eval_data_folder, data_transforms['eval'])
        eval_dataloader = DataLoader(eval_data, self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)

        class_names = train_data.classes
        n_classes = len(class_names)

        return train_dataloader, eval_dataloader, n_classes


class ImageToImageReader(datasets.ImageFolder):
        def __getitem__(self, index):
            img, y = super().__getitem__(index)
            return img, img


class ImageToImageDataLoader(ImageDataLoader):
    def __init__(self,
                 train_data_folder: str, eval_data_folder: str,
                 pil_transformations: list = None,
                 tensor_transformations: list = None,
                 batch_size=64,
                 shuffle=True,
                 num_workers=4,
                 imagenet_format=True,
                 size=None, mean: list = None, std: list = None,
                 apply_transforms_to_eval: bool = False):

        super().__init__(
            train_data_folder, eval_data_folder,
            pil_transformations,
            tensor_transformations,
            batch_size,
            shuffle,
            num_workers,
            imagenet_format,
            size,
            mean,
            std,
            apply_transforms_to_eval,
            dataset_class=ImageToImageReader
        )

    def prepare(self):
        train_dataloader, eval_dataloader, _ = super().prepare()
        return train_dataloader, eval_dataloader
