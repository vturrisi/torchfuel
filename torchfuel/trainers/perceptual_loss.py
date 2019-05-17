import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16

from torchfuel.trainers.base import GenericTrainer
from torchfuel.models.vgg import SlicedVGG


class PerceptualLossTrainer(GenericTrainer):
    """
    Perceptual loss trainer that uses a vgg16 to compute the loss.

    Args:
        - device: torch device
        - model: model to train
        - optimiser: torch optimiser
        - scheduler: learning rate scheduler
        - model_name: name of the trained model
        - print_perf: whether to print performance during training
        - use_avg_loss: whether to use average loss instead of total loss
        - use_tqdm: whether to use tqdm for better visualisation during each step

    """

    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 checkpoint_model: bool = False,
                 checkpoint_every_n: int = 1,
                 model_name: str = 'model.pt',
                 print_perf: bool = True,
                 use_avg_loss: bool = False,
                 use_tqdm: bool = True,
                 ):

        super().__init__(
            device,
            model,
            optimiser,
            scheduler,
            checkpoint_model=checkpoint_model,
            checkpoint_every_n=checkpoint_every_n,
            model_name=model_name,
            print_perf=print_perf,
            use_avg_loss=use_avg_loss,
            use_tqdm=use_tqdm,
        )

        self.vgg = SlicedVGG().to(device)

    def compute_loss(self, output: torch.Tensor, y: torch.Tensor):
        features_out = self.vgg(output)
        features_y = self.vgg(y)

        loss = F.mse_loss(features_out.relu4, features_y.relu4)
        return loss

