import cv2
import numpy as np

from app.config import DIFFUSE_KERNEL_SIZE, DIFFUSE_SIGMA
from app.shaders import BaseShader


class Diffuse(BaseShader):

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = cv2.GaussianBlur(image, (DIFFUSE_KERNEL_SIZE, DIFFUSE_KERNEL_SIZE), sigmaX=DIFFUSE_SIGMA)
        return image
