import torch


def compute_minibatch_loss(trainer):
    data = trainer.state.current_minibatch
    loss = data['loss']

    stats = trainer.state.current_minibatch_stats
    stats['loss'] = loss


def compute_minibatch_correct_preds(trainer):
    data = trainer.state.current_minibatch
    X = data['X']
    y = data['y']
    output = data['output']

    _, pred = torch.max(output, 1)
    correct = torch.sum(pred == y).item()
    batch_size = X.size(0)
    stats = trainer.state.current_minibatch_stats
    stats['size'] = batch_size
    stats['correct_predictions'] = correct


def compute_minibatch_cm(trainer):
    data = trainer.state.current_minibatch
    y = data['y']
    output = data['output']

    _, pred = torch.max(output, 1)
    cm = torch.zeros((trainer.n_classes, trainer.n_classes))
    for p, y_ in zip(pred, y):
        cm[y_, p] += 1
    stats = trainer.state.current_minibatch_stats
    stats['confusion_matrix'] = cm


def compute_epoch_loss(trainer):
    train_loss = 0
    eval_loss = 0

    for s in trainer.state.train.minibatch_stats:
        train_loss += s['loss']

    for s in trainer.state.eval.minibatch_stats:
        eval_loss += s['loss']

    trainer.state.train_loss = train_loss
    trainer.state.eval_loss = eval_loss


def compute_epoch_acc(trainer):
    n = 0
    correct_predictions = 0
    for s in trainer.state.train.minibatch_stats:
        correct_predictions += s['correct_predictions']
        n += s['size']
    train_acc = correct_predictions / n

    n = 0
    correct_predictions = 0
    for s in trainer.state.eval.minibatch_stats:
        correct_predictions += s['correct_predictions']
        n += s['size']
    eval_acc = correct_predictions / n

    trainer.state.train_acc = train_acc
    trainer.state.eval_acc = eval_acc


def compute_epoch_cm(trainer):
    trainer.state.train_cm = sum((s['confusion_matrix']
                                  for s in trainer.state.train.minibatch_stats))

    trainer.state.eval_cm = sum((s['confusion_matrix']
                                 for s in trainer.state.eval.minibatch_stats))
