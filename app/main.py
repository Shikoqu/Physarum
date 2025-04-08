import numpy as np
import time

from app.config import IMAGE_PATH
from app.engine import Engine
from app.particles.particle_system import ParticleSystem
from app.processing.pipeline import Pipeline
from app.processing.shaders import (
    Decay,
    Diffuse,
    DiffuseAndDecay,
    ParticleUpdateShader,
    PheromoneDepositShader,
)


def get_pipeline(
        particle_system: ParticleSystem,
        food_bitmap: np.ndarray,
        noise: np.ndarray,
        sensors_bitmap: np.ndarray,
        particles_bitmap: np.ndarray,
) -> Pipeline:
    pipeline = Pipeline()
    
    pipeline += PheromoneDepositShader(particle_system)
    pipeline += ParticleUpdateShader(particle_system, food_bitmap, noise, sensors_bitmap, particles_bitmap)
    pipeline += Diffuse()
    pipeline += Decay()
    # pipeline += DiffuseAndDecay()

    return pipeline


def main():
    rng = np.random.default_rng(seed=int(time.time()))
    # image = cv2.imread(IMAGE_PATH)
    # shape = image.shape[:2]
    shape = (2560, 1440)

    # food_bitmap = image[:, :, 2].copy()  # Use the blue channel as food
    food_bitmap = np.zeros(shape, dtype=np.uint8)
    noise = rng.random(shape, dtype=np.float32)
    pheromone_bitmap = np.zeros(shape, dtype=np.uint8)
    sensors_bitmap = np.zeros(shape, dtype=np.uint8)
    particles_bitmap = np.zeros(shape, dtype=np.uint8)

    particle_system = ParticleSystem(shape)

    pipeline = get_pipeline(particle_system, food_bitmap, noise, sensors_bitmap, particles_bitmap)

    pheromone_bitmap[-14:, -4:] = 255

    # init and run pygame
    engine = Engine(pheromone_bitmap, pipeline, sensors_bitmap, particles_bitmap)
    engine.init_pygame()
    engine.run()


if __name__ == "__main__":
    main()
