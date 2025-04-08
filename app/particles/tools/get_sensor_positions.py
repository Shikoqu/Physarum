from numba import njit, prange
import numpy as np

from app.utils.time_it import time_it


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

# time_it_configure(__name__)

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
        sensor_offsets: np.ndarray,
) -> np.ndarray:
    num_particles = positions.shape[0]
    sensor_positions = np.zeros((num_particles, 3, 2))
    for i in prange(num_particles):
        angle = angles[i]
        pos = positions[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for j in prange(3):
            dx = sensor_offsets[j, 0]
            dy = sensor_offsets[j, 1]
            sensor_positions[i, j] = pos + np.array(
                [dx * sin_a + dy * cos_a, dx * cos_a - dy * sin_a]
            )

    return sensor_positions


def _get_sensor_positions2(
        angles: np.ndarray,
        positions: np.ndarray,
        sensor_offsets: np.ndarray
) -> np.ndarray:
    cos_a = np.cos(angles)[:, np.newaxis]
    sin_a = np.sin(angles)[:, np.newaxis]

    return positions[:, np.newaxis, :] + np.stack(
        [
            sensor_offsets[:, 0] * sin_a + sensor_offsets[:, 1] * cos_a,
            sensor_offsets[:, 0] * cos_a - sensor_offsets[:, 1] * sin_a,
        ],
        axis=2,
    )
