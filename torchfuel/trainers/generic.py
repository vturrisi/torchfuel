import pickle
import time
from abc import abstractmethod
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchfuel.trainers.const as const
from torchfuel.trainers.generic_hooks import (compute_epoch_time,
                                              log_start_time,
                                              step_on_plateau_scheduler,
                                              step_scheduler)
from torchfuel.trainers.metrics import compute_epoch_loss
from torchfuel.trainers.state import State
from torchfuel.utils.time_parser import parse_seconds
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

        self.state = State()

        self._hooks = {c: list() for c in [const.AFTER_EPOCH, const.BEFORE_EPOCH,
                                           const.AFTER_MINIBATCH, const.BEFORE_MINIBATCH,
                                           const.AFTER_TRAIN, const.BEFORE_TRAIN,
                                           const.AFTER_EVAL, const.BEFORE_EVAL]}

        self._hooks[const.BEFORE_EPOCH].append(log_start_time)
        self._hooks[const.AFTER_EPOCH].append(compute_epoch_loss)
        self._hooks[const.AFTER_EPOCH].append(compute_epoch_time)

        self.print_perf = print_perf
        if print_perf:
            self.print_template = ()

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self._hooks[const.AFTER_EPOCH].append(step_on_plateau_scheduler)
        else:
            self._hooks[const.BEFORE_EPOCH].append(step_scheduler)

    @abstractmethod
    def compute_loss(self, output, y):
        pass

    def execute_on(self, where):
        def wrapper(func):
            if where in self._hooks:
                self._hooks[where].append(func)
            else:
                raise ValueError('Hook {} does not exists'.format(where))
            return func
        return wrapper

    def run_hooks(self, where):
        if where in self._hooks:
            for hook in self._hooks[where]:
                hook(self)
        else:
            raise ValueError('Hook {} does not exists'.format(where))

    def compute_minibatch_statistics(self, X, y, output, loss):
        return {'loss': loss}

    def print_epoch_performance(self, epoch, train_epoch_stats, eval_epoch_stats):
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss
        elapsed_time = self.state.general.elapsed_time

        s = ('(Epoch #{}) Train loss {:.3f}'
             ' | Eval loss {:.4f} ({:.2f} s)')

        s = s.format(epoch, train_loss,
                     eval_loss, elapsed_time)
        print(s)

    def update_best_model(self, best_model, eval_epoch_stats):
        eval_loss = self.state.eval_loss
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
        self.state.train.minibatch_stats = []

        for X, y in tqdm(dataloader, desc=msg, leave=False):
            X = X.to(self.device)
            y = y.to(self.device)

            output, loss = self.train_minibatch(X, y)
            minibatch_stats = self.compute_minibatch_statistics(X, y, output, loss)
            self.state.train.minibatch_stats.append(minibatch_stats)

    def eval_epoch(self, dataloader, epoch):
        self.model.eval()

        msg = 'Evaluating (epoch {})'.format(epoch)
        self.state.eval.minibatch_stats = []

        with torch.set_grad_enabled(False):
            for X, y in tqdm(dataloader, desc=msg, leave=False):
                X = X.to(self.device)
                y = y.to(self.device)

                output, loss = self.eval_minibatch(X, y)
                minibatch_stats = self.compute_minibatch_statistics(X, y, output, loss)
                self.state.eval.minibatch_stats.append(minibatch_stats)

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

        for epoch in range(start_epoch, epochs):
            # run hooks before epoch
            self.run_hooks(const.BEFORE_EPOCH)

            # run hooks before train
            self.run_hooks(const.BEFORE_TRAIN)

            train_epoch_stats = self.train_epoch(train_dataloader, epoch)

            # run hooks after train
            self.run_hooks(const.AFTER_TRAIN)

            # run hooks before eval
            self.run_hooks(const.BEFORE_EVAL)

            eval_epoch_stats = self.eval_epoch(eval_dataloader, epoch)

            # run hooks after eval
            self.run_hooks(const.AFTER_EVAL)

            # run hooks after epoch
            self.run_hooks(const.AFTER_EPOCH)

            best_model = self.update_best_model(best_model, eval_epoch_stats)

            if self.print_perf:
                self.print_epoch_performance(epoch, train_epoch_stats, eval_epoch_stats)

            if epoch != 0:
                self.save_model(epoch, best_model)

        end_time = time.time()
        elapsed_time = end_time - self.state.general.start_time
        days, hours, minutes, seconds = parse_seconds(elapsed_time)
        print('Training done in {}d, {}h {}min {:.2f}s'.format(days, hours, minutes, seconds))

        return self.model.load_state_dict(best_model['model'])
