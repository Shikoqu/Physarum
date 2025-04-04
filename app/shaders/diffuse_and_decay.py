import cv2
import numpy as np

from app.shaders import BaseShader


class DiffuseAndDecay(BaseShader):

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[1, 2, 1],[2, 3, 2],[1, 2, 1]], np.float32) / 16
        image = cv2.filter2D(image, -1, kernel)
        return image
