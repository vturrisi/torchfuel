import pickle
import time
from abc import abstractmethod
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torchfuel.trainers.const as const
from torchfuel.trainers.generic_hooks import (compute_epoch_time,
                                              log_start_time,
                                              step_on_plateau_scheduler,
                                              step_scheduler)
from torchfuel.trainers.metrics import compute_minibatch_loss, compute_epoch_loss
from torchfuel.trainers.state import State
from torchfuel.utils.time_parser import parse_seconds


class GenericTrainer:
    def __init__(self, device, model, optimiser, scheduler,
                 model_name='model.pt', print_perf=True):
        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.model_name = model_name

        self.state = State()

        self._hooks = {c: list() for c in [const.BEFORE_EPOCH, const.AFTER_EPOCH,
                                           const.BEFORE_MINIBATCH, const.AFTER_MINIBATCH,
                                           const.BEFORE_TRAIN_MINIBATCH, const.AFTER_TRAIN_MINIBATCH,
                                           const.BEFORE_EVAL_MINIBATCH, const.AFTER_EVAL_MINIBATCH,
                                           const.BEFORE_TRAIN, const.AFTER_TRAIN,
                                           const.BEFORE_EVAL, const.AFTER_EVAL
                                           ]}

        self._add_hook(log_start_time, const.BEFORE_EPOCH)
        self._add_hook(compute_epoch_loss, const.AFTER_EPOCH)
        self._add_hook(compute_epoch_time, const.AFTER_EPOCH)
        self._add_hook(compute_minibatch_loss, const.AFTER_MINIBATCH)

        self.print_perf = print_perf
        if print_perf:
            self.print_template = ()

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self._add_hook(step_on_plateau_scheduler, const.AFTER_EPOCH)
        else:
            self._add_hook(step_scheduler, const.BEFORE_EPOCH)

    @abstractmethod
    def compute_loss(self, output, y):
        pass

    def execute_on(self, where, every_n_epochs=1):
        def wrapper(func):
            if where in self._hooks:
                self._hooks[where].append((every_n_epochs, func))
            else:
                raise ValueError('Hook {} does not exists'.format(where))
            return func
        return wrapper

    def _add_hook(self, func, where, every_n_epochs=1):
        if where in self._hooks:
            self._hooks[where].append((every_n_epochs, func))
        else:
            raise ValueError('Hook {} does not exists'.format(where))

    def run_hooks(self, where):
        if where in self._hooks:
            for every, hook in self._hooks[where]:
                hook(self)
        else:
            raise ValueError('Hook {} does not exists'.format(where))

    def print_epoch_performance(self, epoch, train_epoch_stats, eval_epoch_stats):
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss
        elapsed_time = self.state.elapsed_time

        s = ('(Epoch #{}) Train loss {:.3f}'
             ' | Eval loss {:.4f} ({:.2f} s)')

        s = s.format(epoch, train_loss,
                     eval_loss, elapsed_time)
        print(s)

    def _update_best_model(self, best_model, eval_epoch_stats):
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

            self.state.current_minibatch_stats = {}
            self.state.current_minibatch = {
                'X': X,
                'y': y
            }

            self.run_hooks(const.BEFORE_MINIBATCH)
            self.run_hooks(const.BEFORE_TRAIN_MINIBATCH)

            output, loss = self.train_minibatch(X, y)

            self.state.current_minibatch.update({
                'output': output,
                'loss': loss,
            })

            self.run_hooks(const.AFTER_MINIBATCH)
            self.run_hooks(const.AFTER_TRAIN_MINIBATCH)

            minibatch_stats = self.state.current_minibatch_stats
            self.state.train.minibatch_stats.append(minibatch_stats)

    def eval_epoch(self, dataloader, epoch):
        self.model.eval()

        msg = 'Evaluating (epoch {})'.format(epoch)
        self.state.eval.minibatch_stats = []

        with torch.set_grad_enabled(False):
            for X, y in tqdm(dataloader, desc=msg, leave=False):
                X = X.to(self.device)
                y = y.to(self.device)

                self.state.current_minibatch_stats = {}
                self.state.current_minibatch = {
                    'X': X,
                    'y': y
                }

                self.run_hooks(const.BEFORE_MINIBATCH)
                self.run_hooks(const.BEFORE_EVAL_MINIBATCH)

                output, loss = self.eval_minibatch(X, y)

                self.state.current_minibatch.update({
                    'output': output,
                    'loss': loss,
                })

                self.run_hooks(const.AFTER_MINIBATCH)
                self.run_hooks(const.AFTER_EVAL_MINIBATCH)

                minibatch_stats = self.state.current_minibatch_stats
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
        # model changed
        except RuntimeError:
            raise
        # model does not exists
        except:
            start_epoch = 0
            best_model = None

        for epoch in range(start_epoch, epochs):
            self.run_hooks(const.BEFORE_EPOCH)

            # train step
            self.run_hooks(const.BEFORE_TRAIN)
            train_epoch_stats = self.train_epoch(train_dataloader, epoch)
            self.run_hooks(const.AFTER_TRAIN)

            # eval step
            self.run_hooks(const.BEFORE_EVAL)
            eval_epoch_stats = self.eval_epoch(eval_dataloader, epoch)
            self.run_hooks(const.AFTER_EVAL)

            self.run_hooks(const.AFTER_EPOCH)

            best_model = self._update_best_model(best_model, eval_epoch_stats)

            if self.print_perf:
                self.print_epoch_performance(epoch, train_epoch_stats, eval_epoch_stats)

            if epoch != 0:
                self.save_model(epoch, best_model)

        end_time = time.time()
        elapsed_time = end_time - self.state.start_time
        days, hours, minutes, seconds = parse_seconds(elapsed_time)
        print('Training done in {}d, {}h {}min {:.2f}s'.format(days, hours, minutes, seconds))

        return self.model.load_state_dict(best_model['model'])
