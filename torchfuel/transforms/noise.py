import torch


class GaussianNoiser():
    def __init__(self, noise_amount):
        self.noise_amount = noise_amount

    def __call__(self, img):
        device = img.device
        noisy_img = img.clone()

        # gaussian noise
        noise = torch.rand(noisy_img.size()).to(device)

        noisy_img += self.noise_amount * noise

        # clamp between 0 and 1
        noisy_img.clamp_(0, 1)

        return noisy_img


class DropPixelNoiser():
    def __init__(self, noise_chance):
        self.noise_chance = noise_chance

    def __call__(self, img):
        noisy_img = img.clone()
        # noise described in the denoising autoencoder paper from bengio
        for i in range(img.size(1)):
            for j in range(img.size(2)):
                if torch.rand(1).item() < self.noise_chance:
                    img[:, i, j] = 0
        return noisy_img
