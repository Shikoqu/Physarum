from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Particle:
    position: np.ndarray
    direction: float

    def __init__(self):
        self.position = np.array([0, 0], dtype=np.float32)
        self.direction = 0

    def __post_init__(self):
        # Ensure angles are in range [0, 2π)
        self.direction = self.direction % (2 * np.pi)

