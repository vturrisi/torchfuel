import time


def log_start_time(state):
    state.general.start_time = time.time()


def compute_epoch_time(state):
    end_time = time.time()
    state.general.elapsed_time = end_time - state.general.start_time
