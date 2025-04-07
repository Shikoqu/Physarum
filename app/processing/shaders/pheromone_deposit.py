import numpy as np

from app.particles.particle_system import ParticleSystem
from app.processing.shaders import BaseShader


class PheromoneDepositShader(BaseShader):
    def __init__(self, particle_system: ParticleSystem):
        self.particle_system = particle_system

    def apply(self, bitmap: np.ndarray) -> None:
        self.particle_system.deposit_pheromone(bitmap)