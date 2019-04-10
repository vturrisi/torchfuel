import os
from contextlib import suppress

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image as tensor_to_pil
from tqdm import tqdm

from torchfuel.data_loaders.image import ImageFolderWithPaths
from torchfuel.layers.utils import Flatten


class CAMResnet(nn.Module):
    def __init__(self, base_resnet, n_classes):
        super().__init__()
        self.activations = nn.Sequential(*list(base_resnet.children())[:-3])
        last_sublayer = list(self.activations[-1][-1].children())[-1]
        if isinstance(last_sublayer, nn.BatchNorm2d):
            n_filters = last_sublayer.num_features
        elif isinstance(last_sublayer, nn.Conv2d):
            n_filters = last_sublayer.out_channels
        else:
            last_sublayer = list(self.activations[-1][-1].children())[-2]
            n_filters = last_sublayer.num_features

        self.gap = nn.Sequential(nn.AvgPool2d(14, 14),
                                 Flatten())
        self.fc_layer = nn.Linear(n_filters, n_classes, bias=False)

        self.activations.requires_grad = False
        self.gap.requires_grad = False

    def forward(self, imgs):
        output = self.activations(imgs)
        output = self.gap(output)
        output = self.fc_layer(output)
        return output

    def gen_cams(self, device, inp_folder, out_folder,
                 minmax=True, absolute=False,
                 imagenet=True, size=None, mean=None, std=None):
        # create transforms for dataset
        t = []
        if imagenet:
            size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            t.append(transforms.Resize(size))
            normalise = True
        if size is not None:
            t.append(transforms.Resize(size))
            if all(v is not None for v in [mean, std]):
                normalise = True
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
                label_name = os.path.dirname(inp_img).split(os.path.sep)[-1]
                folder = os.path.join(out_folder, label_name)
                # make output folder if it
                with suppress(FileExistsError):
                    os.makedirs(folder)

                fname = os.path.basename(inp_img)
                out_name = os.path.join(folder, fname)

                img = img.view(1, *img.size())

                activation_maps = self.activations(img).detach()
                b, c, h, w = activation_maps.size()
                activation_maps = activation_maps.view(c, h, w)
                weights = self.fc_layer.weight[label].detach().view(-1, 1, 1)
                activation_maps = activation_maps * weights
                cam = torch.sum(activation_maps, 0)
                *_, i, j = cam.size()
                cam = cam.view(i, j)

                if minmax:
                    min_v = torch.min(cam)
                    max_v = torch.max(cam)
                    range_v = max_v - min_v
                    cam = (cam - min_v) / range_v
                elif absolute:
                    cam = torch.abs(cam)
                cam = cam.numpy()

                self._save_cam(inp_img, cam, out_name)

    def _save_cam(self, inp_img, cam, out_name):
        cam = np.uint8(255 * cam)
        img = cv2.imread(inp_img)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(out_name, result)
