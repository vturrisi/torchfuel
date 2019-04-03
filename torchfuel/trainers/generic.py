import time
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class GenericTrainer:
    def __init__(self, device, model, optimiser, scheduler,
                 post_epoch_hooks=None,
                 model_name='model.pt', print_perf=True):
        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.model_name = model_name

        # if post_epoch_hooks is None:
        #     post_epoch_hooks = []
        # self.post_epoch_hooks = post_epoch_hooks

        self.print_perf = print_perf
        if print_perf:
            self.print_template = ()

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler_reduce_on_plateau = True
        else:
            self.scheduler_reduce_on_plateau = False

    @abstractmethod
    def compute_loss(self, output, y):
        pass

    # # TODO
    # def execute_post_epoch_hooks(self, epoch_stats):
    #     for hook in self.post_epoch_hooks:
    #         hook(epoch_stats)

    def compute_epoch_loss(self, epoch_stats):
        loss = 0
        for minibatch_stats in epoch_stats:
            loss += minibatch_stats['loss']
        return loss

    def compute_minibatch_statistics(self, X, y, output, loss):
        return {'loss': loss}

    def print_epoch_performance(self, epoch, elapsed_time,
                                train_epoch_stats, eval_epoch_stats):
        train_loss = self.compute_epoch_loss(train_epoch_stats)
        eval_loss = self.compute_epoch_loss(eval_epoch_stats)

        s = ('(Epoch #{}) Train loss {:.3f}'
             ' | Eval loss {:.4f} ({:.2f} s)')

        s = s.format(epoch, train_loss,
                     eval_loss, elapsed_time)
        print(s)

    def update_best_model(self, best_model, eval_epoch_stats):
        eval_loss = self.compute_epoch_loss(eval_epoch_stats)
        if best_model is None or eval_loss < best_model['loss']:
            best_model = {}
            best_model['loss'] = eval_loss
            best_model['model'] = self.model.state_dict()

        return best_model

    def train_minibatch(self, X, y):
        output = self.model(X)
        loss = self.compute_loss(output, y)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        batch_size = X.size(0)
        total_loss = loss.item() * batch_size
        output = output.detach()
        return output, total_loss

    def eval_minibatch(self, X, y):
        output = self.model(X)
        loss = self.compute_loss(output, y)

        batch_size = X.size(0)
        total_loss = loss.item() * batch_size
        output = output.detach()
        return output, total_loss

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        msg = 'Training (epoch {})'.format(epoch)
        epoch_stats = []

        for X, y in tqdm(dataloader, desc=msg, leave=False):
            X = X.to(self.device)
            y = y.to(self.device)

            output, loss = self.train_minibatch(X, y)
            minibatch_stats = self.compute_minibatch_statistics(X, y, output, loss)
            epoch_stats.append(minibatch_stats)

        return epoch_stats

    def eval_epoch(self, dataloader, epoch):
        self.model.eval()

        msg = 'Evaluating (epoch {})'.format(epoch)
        epoch_stats = []

        with torch.set_grad_enabled(False):
            for X, y in tqdm(dataloader, desc=msg, leave=False):
                X = X.to(self.device)
                y = y.to(self.device)

                output, loss = self.eval_minibatch(X, y)
                minibatch_stats = self.compute_minibatch_statistics(X, y, output, loss)
                epoch_stats.append(minibatch_stats)

        return epoch_stats

    def save_model(self, epoch, best_model):
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimiser_state': self.optimiser.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_model': best_model,
        }, self.model_name)

    def load_model(self):
        checkpoint = torch.load(self.model_name)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch']
        best_model = checkpoint['best_model']

        return start_epoch, best_model

    def fit(self, epochs, train_dataloader, eval_dataloader):
        try:
            start_epoch, best_model = self.load_model()
            # file does not exist or pytorch error (model architecture changed)
        except:
            start_epoch = 0
            best_model = None

        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            if not self.scheduler_reduce_on_plateau:
                self.scheduler.step()

            train_epoch_stats = self.train_epoch(train_dataloader, epoch)
            eval_epoch_stats = self.eval_epoch(eval_dataloader, epoch)

            if self.scheduler_reduce_on_plateau:
                eval_loss = self.compute_epoch_loss(eval_epoch_stats)
                self.scheduler.step(eval_loss)

            best_model = self.update_best_model(best_model, eval_epoch_stats)

            end_time = time.time()
            elapsed_time = end_time - start_time

            if self.print_perf:
                self.print_epoch_performance(epoch, elapsed_time,
                                             train_epoch_stats, eval_epoch_stats)

            if epoch != 0:
                self.save_model(epoch, best_model)

        end_time = time.time()
        elapsed_time = end_time - start_time
        days, elapsed_time = divmod(elapsed_time, 86400)
        hours, elapsed_time = divmod(elapsed_time, 3600)
        minutes, elapsed_time = divmod(elapsed_time, 60)
        print('Training done in {}d, {}h {}min {:.2f}s'.format(days, hours, minutes, elapsed_time))

        return self.model.load_state_dict(best_model['model'])
