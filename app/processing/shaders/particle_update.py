from typing import override

import numpy as np

from app.particles.particle_system import ParticleSystem
from app.processing.shaders import BaseShader

PI = np.pi
PI2 = 2 * np.pi

class ParticleUpdateShader(BaseShader):
    def __init__(
            self,
            particle_system: ParticleSystem,
            food_bitmap: np.ndarray,
            noise: np.ndarray,
            sensors_bitmap: np.ndarray,
            particles_bitmap: np.ndarray,
    ):
        self.particle_system = particle_system
        self.food_bitmap = food_bitmap
        self.noise = noise
        self.sensors_bitmap = sensors_bitmap
        self.particles_bitmap = particles_bitmap

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        self.particle_system.update_angles(
            bitmap,
            self.food_bitmap,
            self.noise,
            self.sensors_bitmap,
            self.particles_bitmap,
        )
        self.particle_system.update_position()
