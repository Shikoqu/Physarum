import cv2
import numpy as np
import time

from app.config import IMAGE_PATH
from app.engine import Engine
from app.particles.particle_system import ParticleSystem
from app.processing.pipeline import Pipeline
from app.processing.shaders import (
    Decay,
    Diffuse,
    ParticleUpdateShader,
    PheromoneDepositShader, DiffuseAndDecay,
)


def get_pipeline(
        particle_system: ParticleSystem,
        food_bitmap: np.ndarray,
        noise_bitmap: np.ndarray
) -> Pipeline:
    pipeline = Pipeline()
    
    pipeline += PheromoneDepositShader(particle_system)
    pipeline += ParticleUpdateShader(particle_system, food_bitmap, noise_bitmap)
    # pipeline += Decay()
    # pipeline += Diffuse()
    pipeline += DiffuseAndDecay()

    return pipeline


def main():
    rng = np.random.default_rng(seed=int(time.time()))
    image = cv2.imread(IMAGE_PATH)
    shape = image.shape[:2]

    food_bitmap = image[:, :, 2].copy()  # Use the blue channel as food
    noise_bitmap = rng.integers(0, 255, shape, dtype=np.uint8)
    pheromone_bitmap = np.zeros_like(food_bitmap)

    particle_system = ParticleSystem(shape)

    pipeline = get_pipeline(particle_system, food_bitmap, noise_bitmap)

    # init and run pygame
    engine = Engine(pheromone_bitmap, pipeline)
    engine.init_pygame()
    engine.run()


if __name__ == "__main__":
    main()
