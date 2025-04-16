import numpy as np


# main
IMAGE_PATH: str = "./app/assets/poland.bmp"
FPS_LIMIT: float = 60

# shaders
DECAY_VALUE: int = 1
DIFFUSE_KERNEL_SIZE: int = 3
DIFFUSE_SIGMA: float = 0.0

# particles
NUM_PARTICLES: np.uint32 = np.uint32(360_000)
SENSOR_ANGLE: np.float32 = np.float32(np.pi / 8)
SENSOR_DISTANCE: np.uint8 = np.uint8(8)

MAX_TURN_ANGLE: np.float32 = np.float32(np.pi / 16)
STEP_SIZE: np.float32 = np.float32(1)
