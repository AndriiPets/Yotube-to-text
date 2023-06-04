import time
from functools import wraps


def timeit(func):
    wraps(func)

    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}...")
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"done in {toc - tic:0.4f} seconds")
        return result

    return wrapper
