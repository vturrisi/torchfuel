import pickle
import time
import typing
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchfuel.trainers.const as const
from torchfuel.trainers.hooks.generic import (compute_epoch_time,
                                              log_start_time,
                                              step_on_plateau_scheduler,
                                              step_scheduler)
from torchfuel.trainers.hooks.metrics import (compute_avg_epoch_loss,
                                              compute_epoch_loss)
from torchfuel.trainers.state import State
from torchfuel.utils.time_parser import parse_seconds


class GenericTrainer:
    """
    Implements a generic trainer which provides a train and eval loops,
    basic evaluation metrics (elapsed time and losses), and autosaves models.

    Never use this class directly. First inherit this class and overwrite the compute_loss method.
    This method is used to allow the trainer to know how to compute
    the loss of a model given its output and y.

    update_best_model should also be overwritten if the best model is based on other metrics, e.g., accuracy.

    It is easy to add new functionalities using the execute_on decorator or the add_hook method.
    Functions added by either method should use the constants defined in the const.py file.
    All functions should receive only one parameter which is the trainer.
    Data should be saved on the trainer.state variable (optionally use trainer.state.train/trainer.state.eval)

    Args:
        - device: torch device
        - model: model to train
        - optimiser: torch optimiser
        - scheduler: learning rate scheduler
        - model_name: name of the trained model
        - print_perf: whether to print performance during training
        - use_tqdm: whether to use tqdm for better visualisation during each step
        - use_avg_loss: whether to use average loss instead of total loss

    """

    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 checkpoint_model: bool = False,
                 checkpoint_every_n: int = 1,
                 model_name: str = None,
                 print_perf: bool = True,
                 use_tqdm: bool = True,
                 use_avg_loss: bool = False):

        if checkpoint_model:
            assert checkpoint_every_n > 0
            assert model_name is not None, (
                'When checkpointing the model, model_name should be specified'
            )

        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.checkpoint_model = checkpoint_model
        self.checkpoint_every_n = checkpoint_every_n
        self.model_name = model_name

        self.print_perf = print_perf
        self.use_tqdm = use_tqdm
        self.use_avg_loss = use_avg_loss

        self.state = State()

        # events supported by current hooks
        events = [
            const.BEFORE_EPOCH, const.AFTER_EPOCH,
            const.BEFORE_MINIBATCH, const.AFTER_MINIBATCH,
            const.BEFORE_TRAIN_MINIBATCH, const.AFTER_TRAIN_MINIBATCH,
            const.BEFORE_EVAL_MINIBATCH, const.AFTER_EVAL_MINIBATCH,
            const.BEFORE_TEST_MINIBATCH, const.AFTER_TEST_MINIBATCH,
            const.BEFORE_TRAIN, const.AFTER_TRAIN,
            const.BEFORE_EVAL, const.AFTER_EVAL,
            const.BEFORE_TEST, const.AFTER_TEST,
        ]
        self._hooks: typing.Dict[const.Event, list] = {e: list() for e in events}

        # adds basic hooks to compute elapsed time and losses
        self.add_hook(log_start_time, const.BEFORE_EPOCH)

        if use_avg_loss:
            self.add_hook(compute_avg_epoch_loss, const.AFTER_EPOCH)
        else:
            self.add_hook(compute_epoch_loss, const.AFTER_EPOCH)

        self.add_hook(compute_epoch_time, const.AFTER_EPOCH)

        if scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.add_hook(step_on_plateau_scheduler, const.AFTER_EPOCH)
            else:
                self.add_hook(step_scheduler, const.BEFORE_EPOCH)

    @abstractmethod
    def compute_loss(self, output: torch.Tensor, y: torch.Tensor):
        """
        Abstract method which is used to compute the loss value given the model output and y.

        Args:
            - output: model output
            - y: real y

        """

        pass

    def execute_on(self, where: const.Event, every_n_epochs: int = 1) -> Callable:
        """
        Decorator to add new functions to be executed given an event.
        The only argument passed to the function is the trainer object.
        All data should be saved/read from it using the auxiliary state, state.train and state.eval variables
        Some variables should not have its variables altered, namely:
            - state.current_epoch
            - state.current_minibatch
            - state.train_loss
            - state.eval_loss
            - state.train/eval.minibatch_stats
        However, adding data to the minibatch_stats namespace is fine.


        Args:
            - func: function. First and only argument should be the trainer
            - where: event to trigger the function's execution (use the values defined in const.py)
            - every_n_epochs: number of epochs between executions

        """

        def wrapper(func: typing.Callable):
            if where in self._hooks:
                self._hooks[where].append((every_n_epochs, func))
            else:
                raise ValueError('Hook {} does not exists'.format(where))
            return func
        return wrapper

    def add_hook(self, func: Callable, where: const.Event, every_n_epochs: int = 1) -> None:
        """
        Function to add new functions to be executed given an event.
        The only argument passed to the function is the trainer object.
        All data should be saved/read from it using the auxiliary state, state.train and state.eval variables
        Some variables should not have its variables altered, namely:
            - state.current_epoch
            - state.current_minibatch
            - state.train_loss
            - state.eval_loss
            - state.train/eval.minibatch_stats
        However, adding data to the minibatch_stats namespace is fine.


        Args:
            - func: function. First and only argument should be the trainer
            - where: event to trigger the function's execution (use the values defined in const.py)
            - every_n_epochs: number of epochs between executions

        """

        if where in self._hooks:
            self._hooks[where].append((every_n_epochs, func))
        else:
            raise ValueError('Hook {} does not exists'.format(where))

    def _run_hooks(self, where: const.Event) -> None:
        """
        Execute all functions registered to a given event

        Args:
            - where: event

        """

        if where in self._hooks:
            for every, hook in self._hooks[where]:
                hook(self)
        else:
            raise ValueError('Hook {} does not exists'.format(where))

    def print_epoch_performance(self) -> None:
        """
        Prints the performance of the last epoch.
        Should be overwritten to allow for prettier (or more complex) prints.
        This base function only prints the epoch number, train/eval losses and elapsed time.

        """

        epoch = self.state.current_epoch
        train_loss = self.state.train_loss
        eval_loss = self.state.eval_loss
        elapsed_time = self.state.elapsed_time

        s = ('(Epoch #{}) Train loss {:.3f}'
             ' | Eval loss {:.4f} ({:.2f} s)')

        s = s.format(epoch, train_loss,
                     eval_loss, elapsed_time)
        print(s)

    def update_best_model(
        self,
        best_model: Dict[str, Union[float, Any]]
    ) -> Dict[str, Union[float, Any]]:
        """
        Updates best model using best_model['loss'].
        This is automatically called every epoch.
        Should be overwritten to allow for other comparisons, e.g., using accuracy.

        """

        eval_loss = self.state.eval_loss

        if best_model is None or eval_loss < best_model['loss']:
            best_model = {}
            best_model['loss'] = eval_loss
            best_model['model'] = self.model.state_dict()

        return best_model

    def train_minibatch(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        output = self.model(X)
        loss = self.compute_loss(output, y)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        batch_size = X.size(0)
        total_loss = loss.item() * batch_size
        output = output.detach()
        return output, total_loss

    def eval_minibatch(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        output = self.model(X)
        loss = self.compute_loss(output, y)

        batch_size = X.size(0)
        total_loss = loss.item() * batch_size
        output = output.detach()
        return output, total_loss

    def train_epoch(self, dataloader: DataLoader) -> None:
        self.model.train()

        self.state.train.minibatch_stats = []

        if self.use_tqdm:
            epoch = self.state.current_epoch
            msg = 'Training (epoch {})'.format(epoch)
            it = tqdm(dataloader, desc=msg, leave=False)
        else:
            it = dataloader

        for X, y in it:
            X = X.to(self.device)
            y = y.to(self.device)
            batch_size = X.size(0)

            self.state.current_minibatch_stats = {
                'size': batch_size,
            }

            self.state.current_minibatch = {
                'X': X,
                'y': y,
                'size': batch_size
            }

            self._run_hooks(const.BEFORE_MINIBATCH)
            self._run_hooks(const.BEFORE_TRAIN_MINIBATCH)

            output, loss = self.train_minibatch(X, y)

            self.state.current_minibatch.update({
                'output': output,
            })

            self.state.current_minibatch_stats['loss'] = loss

            self._run_hooks(const.AFTER_MINIBATCH)
            self._run_hooks(const.AFTER_TRAIN_MINIBATCH)

            minibatch_stats = self.state.current_minibatch_stats
            self.state.train.minibatch_stats.append(minibatch_stats)

    def eval_epoch(self, dataloader: DataLoader) -> None:
        self.model.eval()

        self.state.eval.minibatch_stats = []

        if self.use_tqdm:
            epoch = self.state.current_epoch
            msg = 'Evaluating (epoch {})'.format(epoch)
            it = tqdm(dataloader, desc=msg, leave=False)
        else:
            it = dataloader

        with torch.set_grad_enabled(False):
            for X, y in it:
                X = X.to(self.device)
                y = y.to(self.device)
                batch_size = X.size(0)

                self.state.current_minibatch_stats = {
                    'size': batch_size,
                }

                self.state.current_minibatch = {
                    'X': X,
                    'y': y,
                    'size': batch_size
                }

                self._run_hooks(const.BEFORE_MINIBATCH)
                self._run_hooks(const.BEFORE_EVAL_MINIBATCH)

                output, loss = self.eval_minibatch(X, y)

                self.state.current_minibatch.update({
                    'output': output,
                })

                self.state.current_minibatch_stats['loss'] = loss

                self._run_hooks(const.AFTER_MINIBATCH)
                self._run_hooks(const.AFTER_EVAL_MINIBATCH)

                minibatch_stats = self.state.current_minibatch_stats
                self.state.eval.minibatch_stats.append(minibatch_stats)

    def test_epoch(self, dataloader: DataLoader) -> None:
        self.model.eval()

        self.state.test.minibatch_stats = []

        if self.use_tqdm:
            msg = 'Testing'
            it = tqdm(dataloader, desc=msg, leave=False)
        else:
            it = dataloader

        with torch.set_grad_enabled(False):
            for X, y in it:
                X = X.to(self.device)
                y = y.to(self.device)
                batch_size = X.size(0)

                self.state.current_minibatch_stats = {
                    'size': batch_size
                }

                self.state.current_minibatch = {
                    'X': X,
                    'y': y,
                    'size': batch_size
                }

                self._run_hooks(const.BEFORE_MINIBATCH)
                self._run_hooks(const.BEFORE_TEST_MINIBATCH)

                output, loss = self.eval_minibatch(X, y)

                self.state.current_minibatch.update({
                    'output': output,
                })

                self.state.current_minibatch_stats['loss'] = loss

                self._run_hooks(const.AFTER_MINIBATCH)
                self._run_hooks(const.AFTER_TEST_MINIBATCH)

                minibatch_stats = self.state.current_minibatch_stats
                self.state.test.minibatch_stats.append(minibatch_stats)

    def save_model(self, best_model: Dict[str, Union[float, Dict]]) -> None:
        if best_model is not None:
            if self.scheduler is None:
                scheduler_state = None
            else:
                scheduler_state = self.scheduler.state_dict()

            epoch = self.state.current_epoch
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimiser_state': self.optimiser.state_dict(),
                'scheduler_state': scheduler_state,
                'best_model': best_model,
            }, self.model_name)

    def load_model(self) -> Tuple[int, Dict[str, Union[float, Dict]]]:
        checkpoint = torch.load(self.model_name)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        best_model = checkpoint['best_model']

        return start_epoch, best_model

    def load_best_model(self):
        checkpoint = torch.load(self.model_name)
        best_model = checkpoint['best_model']['model']
        self.model.load_state_dict(best_model)

        return self.model

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

        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            self.state.current_epoch = epoch

            self._run_hooks(const.BEFORE_EPOCH)

            # train step
            self._run_hooks(const.BEFORE_TRAIN)
            self.train_epoch(train_dataloader)
            self._run_hooks(const.AFTER_TRAIN)

            # eval step
            self._run_hooks(const.BEFORE_EVAL)
            self.eval_epoch(eval_dataloader)
            self._run_hooks(const.AFTER_EVAL)

            self._run_hooks(const.AFTER_EPOCH)

            best_model = self.update_best_model(best_model)

            if self.print_perf:
                self.print_epoch_performance()

            if self.checkpoint_model and epoch % self.checkpoint_every_n == 0:
                self.save_model(best_model)

        end_time = time.time()
        elapsed_time = end_time - start_time
        days, hours, minutes, seconds = parse_seconds(elapsed_time)
        print('Training done in {}d, {}h {}min {:.2f}s'.format(days, hours, minutes, seconds))

        self.model.load_state_dict(best_model['model'])
        return self.model

    def test(self, test_dataloader, load=False):
        if load:
            self.load_best_model()

        self._run_hooks(const.BEFORE_EPOCH)

        # test step
        self._run_hooks(const.BEFORE_EVAL)
        self.test_epoch(test_dataloader)
        self._run_hooks(const.AFTER_EVAL)

        self._run_hooks(const.AFTER_EPOCH)
