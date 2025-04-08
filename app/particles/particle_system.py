import numpy as np

from app.config import (
    NUM_PARTICLES,
    SENSOR_ANGLE,
    SENSOR_DISTANCE,
)
from app.particles.tools import (
    deposit_pheromone,
    get_sensor_positions,
    sample_sensors,
    update_angles,
    update_positions, sample_particles,
)


class ParticleSystem:
    def __init__(self, bitmap_shape: tuple[int, int]):
        initial_position = np.divide(bitmap_shape, 2)  # center of image

        self.angles = np.linspace(0, 2 * np.pi, NUM_PARTICLES, endpoint=False)
        self.positions = np.full((NUM_PARTICLES, 2), initial_position)
        self.sensor_offsets = self._calculate_sensor_offsets()

        self._bitmap_shape = np.array(bitmap_shape)

    def update_position(self) -> None:
        update_positions(self.positions, self.angles, self._bitmap_shape)

    def deposit_pheromone(self, bitmap: np.ndarray):
        deposit_pheromone(bitmap, self.positions)

    def update_angles(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray,
            noise_bitmap: np.ndarray,
    ) -> None:
        update_angles(
            self.angles,
            sample_particles(noise_bitmap, self.positions),
            *self._sample_sensors(
                pheromone_bitmap,
                food_bitmap,
            ),
        )

    # # # # # # # # # # # # # # # # # # # # # # # # #

    @staticmethod
    def _calculate_sensor_offsets() -> np.ndarray:
        sin_a = np.sin(SENSOR_ANGLE)
        cos_a = np.cos(SENSOR_ANGLE)

        front = [    1,      0]
        left  = [sin_a,  cos_a]
        right = [sin_a, -cos_a]

        return np.array([left, front, right]) * SENSOR_DISTANCE

    def _sample_sensors(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return sample_sensors(pheromone_bitmap, food_bitmap, self._get_sensor_positions())

    def _get_sensor_positions(self) -> np.ndarray:
        return get_sensor_positions(self.angles, self.positions, self.sensor_offsets)
