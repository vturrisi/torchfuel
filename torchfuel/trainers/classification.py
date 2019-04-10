import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchfuel.trainers.const as const
from torchfuel.trainers.generic import GenericTrainer
from torchfuel.trainers.hooks.metrics import (compute_epoch_acc,
                                              compute_epoch_cm,
                                              compute_minibatch_cm,
                                              compute_minibatch_correct_preds)


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
                 n_classes: bool = None,
                 compute_confusion_matrix: bool = False):

        super().__init__(
            device,
            model,
            optimiser,
            scheduler,
            checkpoint_model=checkpoint_model,
            checkpoint_every_n=checkpoint_every_n,
            model_name=model_name,
            print_perf=print_perf
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

    def compute_loss(self, output, y):
        """
        Computes cross-entropy loss between output and y.

        Args:
            - output: model output
            - y: real y

        """

        return F.cross_entropy(output, y)

    def print_epoch_performance(self):
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

        s = ('(Epoch #{}) Train loss {:.3f} & acc {:.2f}'
             ' | Eval loss {:.4f} & acc {:.2f} ({:.2f} s)')

        s = s.format(epoch, train_loss, train_acc,
                     eval_loss, eval_acc, elapsed_time)
        print(s)

    def update_best_model(self, best_model):
        """
        Updates best model using best_model['acc']

        """

        eval_acc = self.state.eval_acc
        if best_model is None or eval_acc > best_model['acc']:
            best_model = {}
            best_model['acc'] = eval_acc
            best_model['model'] = self.model.state_dict()

        return best_model
