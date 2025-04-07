from numba import njit, prange
import numpy as np

from app.time_it import time_it


def get_sensor_positions(
        angles: np.ndarray,
        positions: np.ndarray,
        sensor_offsets: np.ndarray,
) -> np.ndarray:
    args = (
        angles,
        positions,
        sensor_offsets,
    )
    return get_sensor_positions1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@time_it
def get_sensor_positions1(*args):
    """1M particles ~..."""
    return _get_sensor_positions1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #


@njit(parallel=True)
def _get_sensor_positions1(
        positions: np.ndarray,
        angles: np.ndarray,
        sensor_offsets: np.ndarray
) -> np.ndarray:
    sensor_positions = np.zeros((positions.shape[0], 3, 2))

    for i in prange(positions.shape[0]):
        angle = angles[i]
        pos = positions[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for j in prange(3):
            x = sensor_offsets[j, 0]
            y = sensor_offsets[j, 1]
            dx = x * cos_a - y * sin_a
            dy = x * sin_a + y * cos_a

            sensor_positions[i, j] = pos + np.array([dx, dy])

    return sensor_positions