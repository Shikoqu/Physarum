from typing import override

import cv2
import numpy as np
from numba import njit

from app.config import DIFFUSE_KERNEL_SIZE, DIFFUSE_SIGMA
from app.processing.shaders import BaseShader


class Diffuse(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        gaussian_blur(bitmap)


def gaussian_blur(bitmap: np.ndarray) -> None:
    cv2.GaussianBlur(bitmap, (DIFFUSE_KERNEL_SIZE, DIFFUSE_KERNEL_SIZE), sigmaX=DIFFUSE_SIGMA, dst=bitmap)


def kernel_blur(bitmap: np.ndarray) -> None:
    kernel = np.array([[1, 1, 1],
                       [1, 8, 1],
                       [1, 1, 1]]) / 16
    cv2.filter2D(bitmap, -1, kernel, dst=bitmap)
