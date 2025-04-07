import numpy as np


# main
IMAGE_PATH: str = 'assets/furry.png'
FPS_LIMIT: float = 30

# shaders
DECAY_VALUE: np.uint8 = np.uint8(1)
DIFFUSE_KERNEL_SIZE: np.uint8 = np.uint8(3)
DIFFUSE_SIGMA: np.float32 = np.float32(0.0)

# particles
NUM_PARTICLES: np.uint32 = np.uint32(2137)
SENSOR_ANGLE: np.float32 = np.float32(np.pi / 8)
SENSOR_DISTANCE: np.uint8 = np.uint8(3)

MAX_TURN_ANGLE: np.float32 = np.float32(np.pi / 16)
STEP_SIZE: np.uint8 = np.uint8(1.5)
