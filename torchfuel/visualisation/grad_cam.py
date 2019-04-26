import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchfuel.visualisation.visualiser import Visualiser


class GradCAM(Visualiser):
    def __init__(self, model: nn.Module, resolution: int = 14):
        super().__init__()

        assert resolution in [7, 14, 28, 56, 112]
        assert hasattr(model, 'activations')
        assert hasattr(model, 'fc')

        self.resolution = resolution
        self._desired_layer_id = 0
        self._desired_layer_output: torch.Tensor = None
        self._desired_layer_grad: torch.Tensor = None

        self.model = model

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        output = self.model(imgs)
        return output

    def _save_output(self, module, input, output):
        # gets the output of the last layer in which its output
        # is the desired resolution
        batch, channels, h, w = output.size()
        if h == self.resolution:
            self._desired_layer_id = id(module)
            self._desired_layer_output = output.detach()

    def _save_grad(self, module, grad_in, grad_out):
        if self._desired_layer_id == id(module):
            self._desired_layer_grad = grad_in[0].detach()

    def _add_hooks(self):
        self._handlers = []
        for module in self.model.activations.modules():
            self._handlers.append(module.register_forward_hook(self._save_output))
            self._handlers.append(module.register_backward_hook(self._save_grad))

    def _remove_hooks(self):
        for handle in self._handlers:
            handle.remove()
        self._handlers = []

    def _gen_cam(self, img: torch.Tensor) -> torch.Tensor:
        self._add_hooks()

        out = self(img)
        _, pred = torch.max(out, 1)
        out = out[0, pred]
        out.backward()

        self._remove_hooks()

        grads = self._desired_layer_grad
        activation_maps = self._desired_layer_output

        b, c, h, w = activation_maps.size()
        activation_maps = activation_maps.view(c, h, w)
        weights = torch.mean(grads, (2, 3)).view(-1, 1, 1)

        activation_maps = activation_maps * weights
        cam = torch.sum(activation_maps, 0)

        # negative values should be ignored in CAM
        cam = F.relu(cam)

        min_v = torch.min(cam)
        max_v = torch.max(cam)
        range_v = max_v - min_v
        cam = (cam - min_v) / range_v
        cam = cam.cpu().numpy()

        return cam

    def _save_cam_on_image(self, inp_img: str, cam: np.ndarray, out_name: str):
        cam = np.uint8(255 * cam)
        img = cv2.imread(inp_img)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(out_name, result)

    def gen_visualisation(self, img: torch.Tensor, inp_img: str, out_name: str):
        cam = self._gen_cam(img)
        self._save_cam_on_image(inp_img, cam, out_name)
