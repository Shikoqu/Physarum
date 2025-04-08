from numba import njit, prange
import numpy as np

from app.utils.time_it import time_it, time_it_configure


def sample_sensors(
        pheromone_bitmap: np.ndarray,
        food_bitmap: np.ndarray,
        sensor_positions: np.ndarray,
        sensors_bitmap: np.ndarray,
) -> np.ndarray:
    args = (
        pheromone_bitmap,
        food_bitmap,
        sensor_positions,
        sensors_bitmap,
    )
    return sample_sensors1(*args)
    # return sample_sensors2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

# time_it_configure(__name__)

@time_it
def sample_sensors1(*args):
    """1M particles ~1 ms"""
    return _sample_sensors1(*args)

@time_it
def sample_sensors2(*args):
    """1M particles ~12 ms"""
    return _sample_sensors2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _sample_sensors1(
        pheromone_bitmap: np.ndarray,
        food_bitmap: np.ndarray,
        sensor_positions: np.ndarray,
        sensors_bitmap: np.ndarray,
) -> np.ndarray:
    num_particles = sensor_positions.shape[0]
    max_x = food_bitmap.shape[0] - 1
    max_y = food_bitmap.shape[1] - 1
    sensor_values = np.zeros((num_particles, 3), dtype=np.uint8)
    sensors_bitmap[:] = 0

    for i in prange(num_particles):
        for j in prange(3):
            x = int(sensor_positions[i, j, 0])
            y = int(sensor_positions[i, j, 1])

            x = min(max(x, 0), max_x)
            y = min(max(y, 0), max_y)

            sensors_bitmap[x, y] = 255
            sensor_values[i, j] = combine(pheromone_bitmap[x, y], food_bitmap[x, y])

    return sensor_values


def _sample_sensors2(
        pheromone_bitmap: np.ndarray,
        food_bitmap: np.ndarray,
        sensor_positions: np.ndarray
) -> np.ndarray:
    max_x = food_bitmap.shape[0] - 1
    max_y = food_bitmap.shape[1] - 1

    xs = np.clip(sensor_positions[:, :, 0].astype(np.uint32), 0, max_x)
    ys = np.clip(sensor_positions[:, :, 1].astype(np.uint32), 0, max_y)

    return combine(pheromone_bitmap[xs, ys], food_bitmap[xs, ys])

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit
def combine(pheromone: np.ndarray, food: np.ndarray):
    return pheromone + food