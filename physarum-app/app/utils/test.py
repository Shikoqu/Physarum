from numba import njit, prange

from app.utils.time_it import time_it, time_it_configure


def test(n: int) -> None:
    test1(n)
    test2(n)

# # # # # # # # # # # # # # # # # # # # # # # # #

time_it_configure("test")

@time_it
def test1(n: int) -> None:
    return _test1(n)


@time_it
def test2(n: int) -> None:
    return _test2(n)


# # # # # # # # # # # # # # # # # # # # # # # # #


def _test1(n: int) -> None:
    s: str = ""
    for i in range(n):
        s += str(i)


@njit()
def _test2(n: int) -> None:
    s: str = ""
    for i in prange(n):
        s += str(i)

# # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    n = 10_000
    test(n)
