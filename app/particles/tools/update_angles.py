from numba import njit, prange
import numpy as np

from app.config import MAX_TURN_ANGLE
from app.time_it import time_it

def update_angles(
        angles: np.ndarray,
        sensor_values: np.ndarray
):
    args = (
        angles,
        sensor_values,
        MAX_TURN_ANGLE,
    )
    update_angles1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@time_it
def update_angles1(*args):
    """1M particles ~..."""
    _update_angles1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _update_angles1(
        angles: np.ndarray,
        sensor_values: np.ndarray,
        max_turn_angle: float
) -> None:
    pass