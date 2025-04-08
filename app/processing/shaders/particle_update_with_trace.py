from typing import override

import numpy as np

from app.particles.particle_system_with_trace import ParticleSystemWithTrace
from app.processing.shaders import BaseShader


class ParticleUpdateShaderWithTrace(BaseShader):
    def __init__(
            self,
            particle_system: ParticleSystemWithTrace,
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
        self.particle_system.update_angles_with_trace(
            bitmap,
            self.food_bitmap,
            self.noise,
            self.sensors_bitmap,
            self.particles_bitmap,
        )
        self.particle_system.update_position()
