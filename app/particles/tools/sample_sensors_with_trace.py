from numba import njit, prange
import numpy as np


def sample_sensors_with_trace(
        pheromone_bitmap: np.ndarray,
        food_bitmap: np.ndarray,
        sensor_positions: np.ndarray,
        sensors_bitmap: np.ndarray,
) -> np.ndarray:
    return _sample_sensors_with_trace(
        pheromone_bitmap,
        food_bitmap,
        sensor_positions,
        sensors_bitmap,
    )

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _sample_sensors_with_trace(
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

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit
def combine(pheromone: np.ndarray, food: np.ndarray):
    return pheromone + food