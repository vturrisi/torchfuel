import os
import sys
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torchfuel_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(torchfuel_path)

import torchfuel.trainers.const as const

try:
    from generic import GenericTrainer
    from hooks.metrics import (
        compute_epoch_acc,
        compute_epoch_cm,
        compute_minibatch_cm,
        compute_minibatch_correct_preds,
    )
except:
    from .generic import GenericTrainer
    from .hooks.metrics import (
        compute_epoch_acc,
        compute_epoch_cm,
        compute_minibatch_cm,
        compute_minibatch_correct_preds,
    )


class ClassificationTrainer(GenericTrainer):
    """
    Basic classification trainer which uses cross-entropy loss to train a model.
    Also computes accuracy and defines best model based on eval accuracy.

    Args:
        - device: torch device
        - model: model to train
        - optimiser: torch optimiser
        - scheduler: learning rate scheduler
        - model_name: name of the trained model
        - print_perf: whether to print performance during training
        - use_avg_loss: whether to use average loss instead of total loss
        - use_tqdm: whether to use tqdm for better visualisation during each step
        - n_classes: number of classes (optinal, used only when computing confusion matrix)
        - compute_confusion_matrix: whether to compute confusion matrix

    """

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        optimiser: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        n_classes: bool = None,
        compute_confusion_matrix: bool = False,
        checkpoint_model: bool = False,
        checkpoint_every_n: int = 1,
        model_name: str = "model.pt",
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

        # adds hooks to compute correctly predicted instances and accuracy
        self.add_hook(compute_minibatch_correct_preds, const.AFTER_MINIBATCH)
        self.add_hook(compute_epoch_acc, const.AFTER_EPOCH)

        self.compute_confusion_matrix = compute_confusion_matrix
        if compute_confusion_matrix:
            assert n_classes is not None
            self.n_classes = n_classes
            self.add_hook(compute_minibatch_cm, const.AFTER_MINIBATCH)
            self.add_hook(compute_epoch_cm, const.AFTER_EPOCH)

    def compute_loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes cross-entropy loss between output and y.

        Args:
            - output: model output
            - y: real y

        """

        return F.cross_entropy(output, y)

    def print_epoch_performance(self) -> None:
        """
        Prints the performance of the last epoch.
        This is automatically called every epoch.
        This includes: epoch number, train/eval losses, elapsed time and train/eval accuracies.

        """

        epoch = self.state.current_epoch
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss
        elapsed_time = self.state.elapsed_time

        train_acc = self.state.train_acc
        eval_acc = self.state.eval_acc

        s = (
            f"(Epoch #{epoch}) Train loss {train_loss:.3f} & acc {train_acc:.2f}"
            f" | Eval loss {eval_loss:.4f} & acc {eval_acc:.2f} ({elapsed_time:.2f} s)"
        )
        print(s)

    def update_best_model(
        self, best_model: Dict[str, Union[float, Any]]
    ) -> Dict[str, Union[float, Any]]:
        """
        Updates best model using best_model['acc']

        """
        eval_acc = self.state.eval_acc

        if best_model is None or eval_acc > best_model["acc"]:
            best_model = {}
            best_model["acc"] = eval_acc
            best_model["model"] = self.model.state_dict()

        return best_model
