from typing import override

import numpy as np

from app.particles.particle_system import ParticleSystem
from app.processing.shaders import BaseShader


class ParticleUpdateShader(BaseShader):
    def __init__(
            self,
            particle_system: ParticleSystem,
            food_bitmap: np.ndarray,
            noise: np.ndarray,
    ):
        self.particle_system = particle_system
        self.food_bitmap = food_bitmap
        self.noise = noise

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        self.particle_system.update_angles(bitmap, self.food_bitmap,self.noise)
        self.particle_system.update_position()
