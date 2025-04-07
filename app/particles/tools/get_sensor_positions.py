from numba import njit, prange
import numpy as np

from app.time_it import time_it, time_it_configure


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
    # return get_sensor_positions2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

time_it_configure(__name__)

@time_it
def get_sensor_positions1(*args):
    """1M particles ~15 ms"""
    return _get_sensor_positions1(*args)

@time_it
def get_sensor_positions2(*args):
    """1M particles ~50 ms"""
    return _get_sensor_positions2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _get_sensor_positions1(
        angles: np.ndarray,
        positions: np.ndarray,
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


def _get_sensor_positions2(
        angles: np.ndarray,
        positions: np.ndarray,
        sensor_offsets: np.ndarray
) -> np.ndarray:
    cos_a = np.cos(angles)[:, np.newaxis]
    sin_a = np.sin(angles)[:, np.newaxis]

    dx = sensor_offsets[:, 0] * cos_a - sensor_offsets[:, 1] * sin_a
    dy = sensor_offsets[:, 0] * sin_a + sensor_offsets[:, 1] * cos_a
    
    sensor_positions = np.stack([dx, dy], axis=2)
    
    return positions[:, np.newaxis, :] + sensor_positions

    
