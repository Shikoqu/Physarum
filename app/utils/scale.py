import numpy as np
from numba import njit


@njit
def scale_to_01(integer: np.uint8) -> float:
    return integer / 255

