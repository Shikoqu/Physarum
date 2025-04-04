import numpy as np

from app.shaders import BaseShader


class Pipeline:
    def __init__(self):
        self.i_density: np.ndarray = None
        self.i_noise: np.ndarray = None
        self.i_pheromone: np.ndarray = None
        self.shaders: list[BaseShader] = []

    def __iadd__(self, shader: BaseShader) -> None:
        match shader:
            case BaseShader(): self.shaders.append(shader)
            case _: raise TypeError(f"Unsupported shader type: {type(shader)}")
        return self

    def process_frame(self, image: np.ndarray) -> None:
        for shader in self.shaders:
            shader.apply(image)

