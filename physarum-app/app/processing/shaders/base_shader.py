import numpy as np
from abc import ABC, abstractmethod


class BaseShader(ABC):

    @abstractmethod
    def apply(self, bitmap: np.ndarray) -> np.ndarray:
        raise NotImplementedError("abstract method")
