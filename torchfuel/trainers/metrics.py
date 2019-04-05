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
