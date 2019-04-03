import torch


class GaussianNoiser():
    def __init__(self, noise_amount):
        self.noise_amount = noise_amount

    def __call__(self, imgs):
        device = imgs.device
        noisy_imgs = imgs.clone()

        # gaussian noise
        noise = torch.rand(noisy_imgs.size()).to(device)

        noisy_imgs += self.noise_amount * noise

        # clamp between 0 and 1
        noisy_imgs.clamp_(0, 1)

        return noisy_imgs


class DropPixelNoiser():
    def __init__(self, noise_chance):
        self.noise_chance = noise_chance

    def __call__(self, imgs):
        noisy_imgs = imgs.clone()
        # noise described in the denoising autoencoder paper from bengio
        for img in noisy_imgs:
            for i in range(img.size(1)):
                for j in range(img.size(2)):
                    if torch.rand(1).item() < self.noise_chance:
                        img[:, i, j] = 0
        return noisy_imgs
