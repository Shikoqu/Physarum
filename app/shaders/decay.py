import numpy as np

from app.config import DECAY_VALUE
from app.shaders import BaseShader


class Decay(BaseShader):

    def apply(self, image: np.ndarray) -> np.ndarray:
        mask = image > 0
        np.subtract(image, DECAY_VALUE, out=image, where=mask)
        return image
