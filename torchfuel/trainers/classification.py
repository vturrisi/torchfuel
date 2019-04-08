import torch
import torch.nn.functional as F

import torchfuel.trainers.const as const
from torchfuel.trainers.generic import GenericTrainer
from torchfuel.trainers.metrics import (compute_epoch_acc,
                                        compute_minibatch_cm,
                                        compute_minibatch_correct_preds)


class ClassificationTrainer(GenericTrainer):
    def __init__(self, device, model, optimiser, scheduler,
                 model_name='model.pt', print_perf=True,
                 n_classes=None, compute_confusion_matrix=False):
        super().__init__(
            device,
            model,
            optimiser,
            scheduler,
            model_name=model_name,
            print_perf=print_perf
        )

        self._add_hook(compute_epoch_acc, const.AFTER_EPOCH)
        self._add_hook(compute_minibatch_correct_preds, const.AFTER_MINIBATCH)

        self.compute_confusion_matrix = compute_confusion_matrix
        if compute_confusion_matrix:
            assert n_classes is not None
            self.n_classes = n_classes
            self._add_hook(compute_minibatch_cm, const.AFTER_MINIBATCH)

    def compute_loss(self, output, y):
        return F.cross_entropy(output, y)

    def print_epoch_performance(self, epoch, train_epoch_stats, eval_epoch_stats):
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss

        train_acc = self.state.train_acc
        eval_acc = self.state.eval_acc

        elapsed_time = self.state.elapsed_time

        s = ('(Epoch #{}) Train loss {:.3f} & acc {:.2f}'
             ' | Eval loss {:.4f} & acc {:.2f} ({:.2f} s)')

        s = s.format(epoch, train_loss, train_acc,
                     eval_loss, eval_acc, elapsed_time)
        print(s)

    def _update_best_model(self, best_model, eval_epoch_stats):
        eval_acc = self.state.eval_acc
        if best_model is None or eval_acc > best_model['acc']:
            best_model = {}
            best_model['acc'] = eval_acc
            best_model['model'] = self.model.state_dict()

        return best_model
