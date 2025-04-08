from numba import njit, prange
import numpy as np

from app.utils.time_it import time_it


def deposit_pheromone(
        bitmap: np.ndarray,
        positions: np.ndarray,
) -> None:
    args = (
        bitmap,
        positions,
    )
    deposit_pheromone1(*args)
    # deposit_pheromones2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

# time_it_configure(__name__)

@time_it
def deposit_pheromone1(*args):
    """1M particles: ~0.25 ms"""
    _deposit_pheromone1(*args)

@time_it
def deposit_pheromone2(*args):
    """1M particles: ~5 ms"""
    _deposit_pheromone2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _deposit_pheromone1(
        bitmap: np.ndarray,
        positions: np.ndarray,
) -> None:
    for i in prange(positions.shape[0]):
        bitmap[
            int(positions[i, 0]),
            int(positions[i, 1])
        ] = 255

def _deposit_pheromone2(
        bitmap: np.ndarray,
        positions: np.ndarray,
) -> None:
    int_pos = positions.astype(np.uint32)
    bitmap[int_pos[:, 1], int_pos[:, 0]] = 255