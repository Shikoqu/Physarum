from typing import override

import cv2
import numpy as np

from app.processing.shaders import BaseShader


class DiffuseAndDecay(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        kernel = np.array([[1, 2, 1],
                           [2, 3, 2],
                           [1, 2, 1]]) / 16
        bitmap[:] = cv2.filter2D(bitmap, -1, kernel)
