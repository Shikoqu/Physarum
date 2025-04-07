from numba import njit, prange
import numpy as np

from app.time_it import time_it

def sample_sensors(
        bitmap: np.ndarray,
        sensor_positions: np.ndarray
) -> np.ndarray:
    args = (
        bitmap,
        sensor_positions,
    )
    return sample_sensors1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@time_it
def sample_sensors1(*args):
    """1M particles ~..."""
    return _sample_sensors1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _sample_sensors1(
        bitmap: np.ndarray,
        sensor_positions: np.ndarray
) -> np.ndarray:
    num_particles = sensor_positions.shape[0]
    sensor_values = np.zeros((num_particles, 3))
    sensor_positions_scaled = sensor_positions * bitmap.shape

    for i in prange(num_particles):
        for j in prange(3):
            x, y = sensor_positions_scaled[i, j].astype(np.int32)
            x = np.clip(x, 0, bitmap.shape[1] - 1)
            y = np.clip(y, 0, bitmap.shape[0] - 1)
            sensor_values[i, j] = bitmap[y, x]

    return sensor_values