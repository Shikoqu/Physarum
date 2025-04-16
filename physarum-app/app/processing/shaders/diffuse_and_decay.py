from typing import override

import cv2
import numpy as np

from app.processing.shaders import BaseShader

kernel1 = np.array([[1, 2, 1],
                    [2, 3, 2],
                    [1, 2, 1]]) / 16

kernel2 = np.array([[17, 34, 17], [34, 52, 34], [17, 34, 17]]) / 256


class DiffuseAndDecay(BaseShader):

    @override
    def apply(self, bitmap: np.ndarray) -> None:
        global kernel2
        cv2.filter2D(bitmap, -1, kernel2, dst=bitmap)
