from typing import override

import numpy as np

from app.config import DECAY_VALUE
from app.processing.shaders import BaseShader
from app.utils.time_it import time_it, time_it_configure


class Decay(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        mask = bitmap >= DECAY_VALUE
        np.subtract(bitmap, DECAY_VALUE, out=bitmap, where=mask)


def decay(bitmap: np.ndarray) -> None:
    decay1(bitmap)


# # # # # # # # # # # # # # # # # # # # # # # # #

# time_it_configure(__name__)


@time_it
def decay1(bitmap: np.ndarray) -> None:
    """r1M paticles: ... ns"""
    _decay(bitmap)


# # # # # # # # # # # # # # # # # # # # # # # # #


def _decay(bitmap: np.ndarray) -> None:
    mask = bitmap >= DECAY_VALUE
    np.subtract(bitmap, DECAY_VALUE, out=bitmap, where=mask)
