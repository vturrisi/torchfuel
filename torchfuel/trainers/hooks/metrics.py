from contextlib import suppress

import torch


def compute_minibatch_correct_preds(trainer) -> None:
    data = trainer.state.current_minibatch
    X = data['X']
    y = data['y']
    batch_size = data['size']
    output = data['output']

    _, pred = torch.max(output, 1)
    correct = torch.sum(pred == y).item()
    stats = trainer.state.current_minibatch_stats
    stats['size'] = batch_size
    stats['correct_predictions'] = correct


def compute_minibatch_cm(trainer) -> None:
    data = trainer.state.current_minibatch
    y = data['y']
    output = data['output']

    _, pred = torch.max(output, 1)
    cm = torch.zeros((trainer.n_classes, trainer.n_classes))
    for p, y_ in zip(pred, y):
        cm[y_, p] += 1
    stats = trainer.state.current_minibatch_stats
    stats['confusion_matrix'] = cm


def compute_epoch_loss(trainer) -> None:
    with suppress(AttributeError):
        train_loss = 0
        for state in trainer.state.train.minibatch_stats:
            train_loss += state['loss']
        trainer.state.train_loss = train_loss

    with suppress(AttributeError):
        eval_loss = 0
        for state in trainer.state.eval.minibatch_stats:
            eval_loss += state['loss']
        trainer.state.eval_loss = eval_loss

    with suppress(AttributeError):
        test_loss = 0
        for state in trainer.state.test.minibatch_stats:
            test_loss += state['loss']
        trainer.state.test_loss = eval_loss


def compute_avg_epoch_loss(trainer) -> None:
    with suppress(AttributeError):
        n = 0
        train_loss = 0
        for state in trainer.state.train.minibatch_stats:
            train_loss += state['loss']
            n += state['size']
        trainer.state.train_loss = train_loss / n

    with suppress(AttributeError):
        n = 0
        eval_loss = 0
        for state in trainer.state.eval.minibatch_stats:
            eval_loss += state['loss']
            n += state['size']
        trainer.state.eval_loss = eval_loss / n

    with suppress(AttributeError):
        n = 0
        test_loss = 0
        for state in trainer.state.test.minibatch_stats:
            test_loss += state['loss']
            n += state['size']
        trainer.state.test_loss = eval_loss / n


def compute_epoch_acc(trainer) -> None:
    with suppress(AttributeError):
        n = 0
        correct_predictions = 0
        for state in trainer.state.train.minibatch_stats:
            correct_predictions += state['correct_predictions']
            n += state['size']
        train_acc = correct_predictions / n
        trainer.state.train_acc = train_acc

    with suppress(AttributeError):
        n = 0
        correct_predictions = 0
        for state in trainer.state.eval.minibatch_stats:
            correct_predictions += state['correct_predictions']
            n += state['size']
        eval_acc = correct_predictions / n

        trainer.state.eval_acc = eval_acc

    with suppress(AttributeError):
        n = 0
        correct_predictions = 0
        for state in trainer.state.test.minibatch_stats:
            correct_predictions += state['correct_predictions']
            n += state['size']
        test_acc = correct_predictions / n

        trainer.state.test_acc = test_acc


def compute_epoch_cm(trainer) -> None:
    with suppress(AttributeError):
        trainer.state.train_cm = sum((state['confusion_matrix']
                                      for state in trainer.state.train.minibatch_stats))

    with suppress(AttributeError):
        trainer.state.eval_cm = sum((state['confusion_matrix']
                                     for state in trainer.state.eval.minibatch_stats))

    with suppress(AttributeError):
        trainer.state.test_cm = sum((state['confusion_matrix']
                                     for state in trainer.state.test.minibatch_stats))
