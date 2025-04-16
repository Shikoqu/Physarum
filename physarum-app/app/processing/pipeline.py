import numpy as np

from app.processing.shaders import BaseShader


class Pipeline:
    def __init__(self):
        self.shaders: list[BaseShader] = []

    def __iadd__(self, shader: BaseShader) -> 'Pipeline':
        match shader:
            case BaseShader(): self.shaders.append(shader)
            case _: raise TypeError(f"Unsupported shader type: {type(shader)}")
        return self

    def process_frame(self, bitmap: np.ndarray) -> None:
        for shader in self.shaders:
            shader.apply(bitmap)
