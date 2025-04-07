from numba import njit, prange
import numpy as np

from app.config import STEP_SIZE
from app.time_it import time_it, time_it_configure


def update_positions(
        positions: np.ndarray,
        angles: np.ndarray,
        bitmap_shape: np.ndarray,
) -> None:
    args = (
        positions,
        angles,
        bitmap_shape,
        STEP_SIZE,
    )
    update_positions1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

time_it_configure(__name__)

@time_it
def update_positions1(*args):
    """1M particles ~17 ms"""
    _update_positions1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

def _update_positions1(
        positions: np.ndarray,
        angles: np.ndarray,
        bitmap_shape: np.ndarray,
        step_size: np.uint8,
) -> None:
    steps = np.stack([np.sin(angles), np.cos(angles)], axis=1)

    np.clip(
        positions + steps * step_size,
        [0, 0],
        [bitmap_shape[0] - 1, bitmap_shape[1] - 1],
        out=positions,
        )

    horizontal_wall = np.nonzero(
        (positions[:, 0] == 0) | (positions[:, 0] == bitmap_shape[0] - 1)
    )[0]

    vertical_wall = np.nonzero(
        (positions[:, 1] == 0) | (positions[:, 1] == bitmap_shape[1] - 1)
    )[0]

    angles[horizontal_wall] = ((2*np.pi) - angles[horizontal_wall]) % (2*np.pi)
    angles[vertical_wall] = (np.pi - angles[vertical_wall]) % (2*np.pi)


