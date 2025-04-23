import numpy as np

from app.particles.particle_system import ParticleSystem
from app.particles.tools import (
    deposit_pheromone,
    get_sensor_positions,
    update_angles,
    update_positions,
    sample_particles_with_trace,
    sample_sensors_with_trace,
)


class ParticleSystemWithTrace(ParticleSystem):
    def __init__(self, food_bitmap: np.ndarray):
        super().__init__(food_bitmap)

    def update_position(self) -> None:
        update_positions(self.positions, self.angles, self._bitmap_shape)

    def deposit_pheromone(self, bitmap: np.ndarray):
        deposit_pheromone(bitmap, self.positions)

    def update_angles_with_trace(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray,
            noise: np.ndarray,
            sensors_bitmap: np.ndarray,
            particles_bitmap: np.ndarray,
    ) -> None:
        update_angles(
            self.angles,
            sample_particles_with_trace(noise, self.positions, particles_bitmap),
            self._sample_sensors_with_trace(
                pheromone_bitmap,
                food_bitmap,
                sensors_bitmap,
            ),
        )

    def _sample_sensors_with_trace(
            self,
            pheromone_bitmap: np.ndarray,
            food_bitmap: np.ndarray,
            sensors_bitmap: np.ndarray,
    ) -> np.ndarray:
        return sample_sensors_with_trace(pheromone_bitmap, food_bitmap, self._get_sensor_positions(), sensors_bitmap)

    def _get_sensor_positions(self) -> np.ndarray:
        return get_sensor_positions(self.angles, self.positions, self.sensor_offsets)
