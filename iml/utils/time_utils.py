import time


def time_wrapper(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    elapse = time.time() - start
    return res, elapse
