from numba import njit, prange
import numpy as np

from app.config import MAX_TURN_ANGLE
from app.utils import time_it, time_it_configure, scale_to_01


def update_angles(
        angles: np.ndarray,
        noise_values: np.ndarray,
        pheromone_sensor_values: np.ndarray,
        food_sensor_values: np.ndarray,
):
    args = (
        angles,
        noise_values,
        pheromone_sensor_values,
        food_sensor_values,
        MAX_TURN_ANGLE,
    )
    update_angles1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

# time_it_configure(__name__)

@time_it
def update_angles1(*args):
    """1M particles ~..."""
    _update_angles1(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _update_angles1(
        angles: np.ndarray,
        noise_values: np.ndarray,
        pheromone_sensor_values: np.ndarray,
        food_sensor_values: np.ndarray,
        max_turn_angle: float
) -> None:
    combined = combine(pheromone_sensor_values, food_sensor_values)
    num_particles = angles.shape[0]

    for i in prange(num_particles):
        sl = combined[i, 0]
        sf = combined[i, 1]
        sr = combined[i, 2]
        noise = scale_to_01(noise_values[i])
        turn_angle = 0

        if   sl < sf > sr:   # continue forward
            ...
        elif sl > sf < sr:   # turn randomly
            turn_angle = max_turn_angle * (noise - 0.5) * 2
        elif sl < sr:        # turn right
            turn_angle = -max_turn_angle * noise
        elif sl > sr:        # turn left
            turn_angle = max_turn_angle * noise

        angles[i] = (angles[i] + turn_angle) % (2 * np.pi)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit
def combine(pheromone: np.ndarray, food: np.ndarray):
    return pheromone + food