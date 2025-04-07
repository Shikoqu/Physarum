import functools
import time

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


def time_it(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        name = function.__name__
        if is_enabled(name):
            print(f"{name}")
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print(f"\ttook {(end-start)*1000:.3f} ms")
            return result
        else:
            return function(*args, **kwargs)
    return wrapper
