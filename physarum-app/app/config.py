import numpy as np


# main
# IMAGE_PATH: str = "./app/assets/krakow.png"
# IMAGE_NEGATIVE: bool = True

IMAGE_PATH: str = "./app/assets/poland.bmp"
IMAGE_NEGATIVE: bool = False

FPS_LIMIT: float = 60

# shaders
DECAY_VALUE: int = 1
DIFFUSE_KERNEL_SIZE: int = 3
DIFFUSE_SIGMA: float = 0.0

# particles
NUM_PARTICLES: np.uint32 = np.uint32(1_000_000)
SENSOR_ANGLE: np.float32 = np.float32(np.pi / 8)
SENSOR_DISTANCE: np.uint8 = np.uint8(8)

MAX_TURN_ANGLE: np.float32 = np.float32(np.pi / 8)
STEP_SIZE: np.float32 = np.float32(0.5)

# particle distribution
PARTICLE_DISTRIBUTION_METHOD: str = "ring"  # options: "random", "center", "circle", "ring", "grid", "food_based"
PARTICLE_DISTRIBUTION_POINT: tuple[float, float] | None = (0.5, 0.5)
PARTICLE_DISTRIBUTION_FACE_BACK: bool = False
PARTICLE_DISTRIBUTION_DIAMETER: float = 0.9
