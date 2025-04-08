import numpy as np
import time

from app.engine_with_trace import EngineWithTrace
from app.particles.particle_system_with_trace import ParticleSystemWithTrace
from app.processing.pipeline import Pipeline
from app.processing.shaders import (
    Decay,
    Diffuse,
    ParticleUpdateShaderWithTrace,
    PheromoneDepositShader,
)


def get_pipeline_with_trace(
        particle_system: ParticleSystemWithTrace,
        food_bitmap: np.ndarray,
        noise: np.ndarray,
        sensors_bitmap: np.ndarray,
        particles_bitmap: np.ndarray,
) -> Pipeline:
    pipeline = Pipeline()

    pipeline += PheromoneDepositShader(particle_system)
    pipeline += ParticleUpdateShaderWithTrace(particle_system, food_bitmap, noise, sensors_bitmap, particles_bitmap)
    # pipeline += DiffuseAndDecay()
    pipeline += Diffuse()
    pipeline += Decay()

    return pipeline


def main_with_trace():
    rng = np.random.default_rng(seed=int(time.time()))
    shape = (2560, 1440)

    noise = rng.random(shape, dtype=np.float32)
    food_bitmap = np.zeros(shape, dtype=np.uint8)
    sensors_bitmap = np.zeros(shape, dtype=np.uint8)
    pheromone_bitmap = np.zeros(shape, dtype=np.uint8)
    particles_bitmap = np.zeros(shape, dtype=np.uint8)

    particle_system = ParticleSystemWithTrace(shape)

    pipeline = get_pipeline_with_trace(particle_system, food_bitmap, noise, sensors_bitmap, particles_bitmap)

    engine = EngineWithTrace(pipeline, pheromone_bitmap, sensors_bitmap, particles_bitmap)
    engine.init_pygame()
    engine.run()



if __name__ == "__main__":
    main_with_trace()
