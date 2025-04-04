import cv2
import numpy as np
import pygame as pg

from app.config import FPS_LIMIT
from app.processing.pipeline import Pipeline


class Engine:
    def __init__(self, bitmap: np.ndarray, pipeline: Pipeline):
        self.current_bitmap: np.ndarray = bitmap.swapaxes(0, 1).copy()
        self.pipeline: Pipeline = pipeline

        self._shape: tuple[int, int] = bitmap.shape[::-1]
        self._clock: pg.time.Clock = None
        self._is_running: bool = False
        self._window_size: tuple[int, int] = None
        self._display_surface: pg.Surface = None

    def init_pygame(self):
        pg.init()
        pg.display.set_caption("physarum")

        self._window_size = self._shape
        self._display_surface = pg.display.set_mode(self._window_size)
        self._clock = pg.time.Clock()
        self._is_running = True

    def handle_pg_events(self, events):
        for event in events:
            if event.type == pg.QUIT:
                self._is_running = False

    def process(self):
        self.pipeline.process_frame(self.current_bitmap)
        image = cv2.cvtColor(self.current_bitmap, cv2.COLOR_GRAY2RGB)
        surface = pg.surfarray.make_surface(image)
        self._display_surface.blit(surface, (0, 0))
        pg.display.flip()

    def run(self):
        self.init_pygame()
        frame_count = 0
        
        while self._is_running:
            self.handle_pg_events(pg.event.get())
            self.process()

            if frame_count % 30 == 0:
                pg.display.set_caption(f"physarum - FPS: {self._clock.get_fps():.1f}")

            frame_count += 1
            self._clock.tick(FPS_LIMIT)



