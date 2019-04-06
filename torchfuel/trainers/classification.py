import torch
import torch.nn.functional as F

import torchfuel.trainers.const as const
from torchfuel.trainers.generic import GenericTrainer
from torchfuel.trainers.metrics import compute_epoch_acc


class ClassificationTrainer(GenericTrainer):
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

        self._add_hook(compute_epoch_acc, const.AFTER_EPOCH)

    def compute_loss(self, output, y):
        return F.cross_entropy(output, y)

    def print_epoch_performance(self, epoch, train_epoch_stats, eval_epoch_stats):
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss

        train_acc = self.state.train_acc
        eval_acc = self.state.eval_acc

        elapsed_time = self.state.general.elapsed_time

        s = ('(Epoch #{}) Train loss {:.3f} & acc {:.2f}'
             ' | Eval loss {:.4f} & acc {:.2f} ({:.2f} s)')

        s = s.format(epoch, train_loss, train_acc,
                     eval_loss, eval_acc, elapsed_time)
        print(s)

    def _update_best_model(self, best_model, eval_epoch_stats):
        eval_acc = self.state.eval_acc
        if best_model is None or eval_acc < best_model['acc']:
            best_model = {}
            best_model['acc'] = eval_acc
            best_model['model'] = self.model.state_dict()

        return best_model

    def compute_correct_preds(self, output, y):
        _, pred = torch.max(output, 1)
        correct = torch.sum(pred == y).item()
        return correct

    def compute_minibatch_statistics(self, X, y, output, loss):
        d = super().compute_minibatch_statistics(X, y, output, loss)
        correct = self.compute_correct_preds(output, y)
        batch_size = X.size(0)
        d.update({'correct_predictions': correct, 'size': batch_size})
        return d
