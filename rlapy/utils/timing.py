import time


def fast_timer(no_op: bool):
    if no_op:
        quick_time = lambda: 0.0
        return quick_time
    else:
        return time.time
