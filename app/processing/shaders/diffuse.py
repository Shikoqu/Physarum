from typing import override

import cv2
import numpy as np

from app.config import DIFFUSE_KERNEL_SIZE, DIFFUSE_SIGMA
from app.processing.shaders import BaseShader


class Diffuse(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        cv2.GaussianBlur(bitmap, (DIFFUSE_KERNEL_SIZE, DIFFUSE_KERNEL_SIZE), sigmaX=DIFFUSE_SIGMA, dst=bitmap)
