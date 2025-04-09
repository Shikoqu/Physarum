import numpy as np
import time
import cv2

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
) -> Pipeline:
    pipeline = Pipeline()

    pipeline += PheromoneDepositShader(particle_system)
    pipeline += ParticleUpdateShader(particle_system, food_bitmap, noise)
    # pipeline += DiffuseAndDecay()
    pipeline += Diffuse()
    pipeline += Decay()

    return pipeline


def main():
    rng = np.random.default_rng(seed=int(time.time()))
    image = cv2.imread(IMAGE_PATH).swapaxes(0, 1)
    shape = image.shape[:2]

    food_bitmap = image[:, :, 2].copy()  # Use the blue channel as food
    pheromone_bitmap = np.zeros(shape, dtype=np.uint8)
    noise = rng.random(shape, dtype=np.float32)

    particle_system = ParticleSystem(shape)
    pipeline = get_pipeline(particle_system, food_bitmap, noise)

    engine = Engine(pipeline, pheromone_bitmap, food_bitmap)
    engine.init_pygame()
    engine.run()


if __name__ == "__main__":
    main()
