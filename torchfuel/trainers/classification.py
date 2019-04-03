import torch
import torch.nn.functional as F

from torchfuel.trainers.generic import GenericTrainer


class BasicClassificationTrainer(GenericTrainer):
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
        return F.cross_entropy(output, y)

    def compute_epoch_acc(self, epoch_stats):
        n = 0
        correct_predictions = 0
        for minibatch_stats in epoch_stats:
            correct_predictions += minibatch_stats['correct_predictions']
            n += minibatch_stats['minibatch_size']
        acc = correct_predictions / n
        return acc

    def print_epoch_performance(self, epoch, elapsed_time,
                                train_epoch_stats, eval_epoch_stats):
        train_loss = self.compute_epoch_loss(train_epoch_stats)
        train_acc = self.compute_epoch_acc(train_epoch_stats)

        eval_loss = self.compute_epoch_loss(eval_epoch_stats)
        eval_acc = self.compute_epoch_acc(eval_epoch_stats)
        s = ('(Epoch #{}) Train loss {:.3f} & acc {:.2f}'
             ' | Eval loss {:.4f} & acc {:.2f} ({:.2f} s)')

        s = s.format(epoch, train_loss, train_acc,
                     eval_loss, eval_acc, elapsed_time)
        print(s)

    def update_best_model(self, best_model, eval_epoch_stats):
        eval_acc = self.compute_epoch_acc(eval_epoch_stats)
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
        d.update({'correct_predictions': correct, 'minibatch_size': batch_size})
        return d
