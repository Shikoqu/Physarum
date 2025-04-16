from typing import override

import numpy as np
import pygame as pg

from app.engine import Engine
from app.processing.pipeline import Pipeline


class EngineWithTrace(Engine):
    def __init__(
            self,
            pipeline: Pipeline,
            bitmap: np.ndarray,
            sensors_bitmap: np.ndarray,
            particles_bitmap: np.ndarray,
    ):
        super().__init__(pipeline, bitmap)
        self.sensors_bitmap = sensors_bitmap
        self.particles_bitmap = particles_bitmap

    @override
    def process(self):
        self.pipeline.process_frame(self.current_bitmap)
        image = np.stack([
            self.current_bitmap,
            self.particles_bitmap,
            self.sensors_bitmap,
        ], axis=2)
        surface = pg.surfarray.make_surface(image)
        surface = pg.transform.scale(surface, self._window_size)
        self._display_surface.blit(surface, (0, 0))
        pg.display.flip()
