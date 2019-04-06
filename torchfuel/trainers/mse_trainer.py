import torch.nn.functional as F

from torchfuel.trainers.generic import GenericTrainer


class MSETrainer(GenericTrainer):
    def compute_loss(self, output, y):
        return F.mse_loss(output, y)
