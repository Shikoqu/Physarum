import cv2
import numpy as np
from abc import ABC, abstractmethod


class BaseShader(ABC):

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("abstract method")
