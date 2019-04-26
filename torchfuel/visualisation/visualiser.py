
import os
from abc import abstractmethod
from contextlib import suppress
from typing import Iterable, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from torchfuel.data_loaders.image import ImageFolderWithPaths


class Visualiser(nn.Module):
    @abstractmethod
    def gen_visualisation(self, img: torch.Tensor, inp_img: str, out_name: str):
        pass

    def gen_visualisation_for_multiple_images(
        self,
        device: torch.device,
        inp_folder: str,
        out_folder: str,
        imagenet: Optional[bool] = False,
        size: Union[int, Iterable] = None,
        mean: Optional[Iterable] = None,
        std: Optional[Iterable] = None
    ) -> None:
        # create transforms for dataset
        t = []
        if imagenet:
            size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t.append(transforms.Resize(size))
            normalise = True
        elif size is not None:
            t.append(transforms.Resize(size))
            if all(v is not None for v in [mean, std]):
                normalise = True
            else:
                normalise = False
        else:
            raise Exception('Use imagenet or specify at least size')
            normalise = False
        t.append(transforms.ToTensor())
        if normalise:
            t.append(transforms.Normalize(mean, std))
        data_transforms = transforms.Compose(t)

        dataset = ImageFolderWithPaths(inp_folder, data_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

        for paths, imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            for inp_img, img, label in zip(paths, imgs, labels):
                img = img.view(1, *img.size())
                label = label.view(1)

                # make output folder if it does not exists
                with suppress(FileExistsError):
                    os.makedirs(out_folder)

                pred = torch.max(self(img), 1)[1].item()
                fname = os.path.basename(inp_img)
                name, ext = os.path.splitext(fname)
                label_name = os.path.dirname(inp_img).split(os.path.sep)[-1]
                fname = '{}_real={}({})_pred={}{}'.format(name, label.item(), label_name, pred, ext)
                out_name = os.path.join(out_folder, fname)

                self.gen_visualisation(img, inp_img, out_name)
