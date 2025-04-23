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
    update_positions,
    sample_particles,
)
from app.particles.particle_factory import ParticleFactory


class ParticleSystem:
    def __init__(self, food_bitmap: np.ndarray):
        factory = ParticleFactory()
        self.positions, self.angles = factory.prime_particles(food_bitmap)

        self._bitmap_shape = food_bitmap.shape
        self.sensor_offsets = self._calculate_sensor_offsets()


    def update_position(self) -> None:
        update_positions(self.positions, self.angles, self._bitmap_shape)

    def deposit_pheromone(self, bitmap: np.ndarray):
        deposit_pheromone(bitmap, self.positions)

    def update_angles(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray,
            noise: np.ndarray,
    ) -> None:
        update_angles(
            self.angles,
            sample_particles(noise, self.positions),
            self._sample_sensors(pheromone_bitmap, food_bitmap),
        )

    # # # # # # # # # # # # # # # # # # # # # # # # #

    @staticmethod
    def _calculate_sensor_offsets() -> np.ndarray:
        sin_a = np.sin(SENSOR_ANGLE)
        cos_a = np.cos(SENSOR_ANGLE)
        return np.array([
            [cos_a,  sin_a], # left sensor
            [    1,      0], # front sensor
            [cos_a, -sin_a], # right sensor
        ]) * SENSOR_DISTANCE

    def _sample_sensors(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray,
    ) -> np.ndarray:
        return sample_sensors(pheromone_bitmap, food_bitmap, self._get_sensor_positions())

    def _get_sensor_positions(self) -> np.ndarray:
        return get_sensor_positions(self.angles, self.positions, self.sensor_offsets)
