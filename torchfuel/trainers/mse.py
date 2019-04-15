import torch
import torch.nn.functional as F

from torchfuel.trainers.generic import GenericTrainer


class MSETrainer(GenericTrainer):
    def compute_loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, y)
