import functools
import time
from contextlib import contextmanager
from typing import Callable, Generator, Any

ENABLED: bool = True
ENABLED_FUNCTIONS: set[str] = set()


def time_it_configure(name: str):
    global ENABLED_FUNCTIONS
    ENABLED_FUNCTIONS.add(name.split(".")[-1])


def is_enabled(function_name: str):
    if not ENABLED:
        return False

    if function_name in ENABLED_FUNCTIONS:
        return True

    for name in ENABLED_FUNCTIONS:
        if function_name.startswith(name):
            return True

    return False


@contextmanager
def timer(name: str) -> Generator[None, Any, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        print(f"{name}\n\ttook {(end-start)/1000:.2f} ms")


def time_it(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if is_enabled(function.__name__):
            with timer(function.__name__):
                return function(*args, **kwargs)
        else:
            return function(*args, **kwargs)
    return wrapper
