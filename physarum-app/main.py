import numpy as np
import time
import cv2
import os

from app.config import IMAGE_PATH, IMAGE_NEGATIVE
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
    pipeline += DiffuseAndDecay()
    # pipeline += Diffuse()
    pipeline += Decay()

    return pipeline


def get_food_bitmap(color_channel: int) -> np.ndarray:
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if (image := cv2.imread(IMAGE_PATH).swapaxes(0, 1)) is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    bitmap = image[:, :, color_channel].copy()
    return bitmap if not IMAGE_NEGATIVE else 255 - bitmap


def main():
    rng = np.random.default_rng(seed=int(time.time()))
    food_bitmap = get_food_bitmap(color_channel=1)
    shape = food_bitmap.shape

    pheromone_bitmap = np.zeros(shape, dtype=np.uint8)
    noise = rng.random(shape, dtype=np.float32)

    particle_system = ParticleSystem(food_bitmap)
    pipeline = get_pipeline(particle_system, food_bitmap, noise)

    engine = Engine(pipeline, pheromone_bitmap, food_bitmap)
    engine.init_pygame()
    engine.run()


if __name__ == "__main__":
    main()
