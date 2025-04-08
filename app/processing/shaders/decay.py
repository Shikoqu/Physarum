from typing import override

import numpy as np

from app.config import DECAY_VALUE
from app.processing.shaders import BaseShader


class Decay(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        mask = bitmap >= DECAY_VALUE
        np.subtract(bitmap, DECAY_VALUE, out=bitmap, where=mask)
