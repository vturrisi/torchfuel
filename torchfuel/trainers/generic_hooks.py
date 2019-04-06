import time


def log_start_time(trainer):
    trainer.state.start_time = time.time()


def compute_epoch_time(trainer):
    end_time = time.time()
    trainer.state.elapsed_time = end_time - trainer.state.start_time


def step_scheduler(trainer):
    trainer.scheduler.step()


def step_on_plateau_scheduler(trainer):
    trainer.scheduler.step(trainer.state.eval_loss)
