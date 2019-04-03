import torch.nn.functional as F

from torchfuel.trainers.generic import GenericTrainer


class BasicAutoencoderTrainer(GenericTrainer):
    def __init__(self, device, model, optimiser, scheduler,
                 model_name='model.pt', print_perf=True):
        super().__init__(
            device,
            model,
            optimiser,
            scheduler,
            model_name=model_name,
            print_perf=print_perf
        )

    def compute_loss(self, output, y):
        return F.mse_loss(output, y)
