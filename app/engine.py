import cv2
import numpy as np
import pygame as pg

from app.config import FPS_LIMIT
from app.processing.pipeline import Pipeline


class Engine:
    def __init__(
            self,
            pipeline: Pipeline,
            bitmap: np.ndarray,
            food_bitmap: np.ndarray = None,
    ):
        self.current_bitmap: np.ndarray = bitmap
        self.pipeline: Pipeline = pipeline
        self.food_bitmap: np.ndarray | None = food_bitmap

        self._clock: pg.time.Clock | None = None
        self._is_running: bool = False

        self._shape: tuple[int, int] = bitmap.shape
        self._resize_scale: tuple[float, float] = (1., 1.)
        self._window_size: tuple[int, int] | None = None
        self._display_surface: pg.Surface | None = None

    def init_pygame(self):
        pg.init()
        pg.display.set_caption("physarum")

        self._window_size = np.multiply(self._shape, self._resize_scale)
        self._display_surface = pg.display.set_mode(self._window_size, pg.RESIZABLE)
        self._clock = pg.time.Clock()
        self._is_running = True

    def resize_window(self, size: tuple[int, int]):
        scale = np.min(np.divide(size, self._shape))
        self._resize_scale = (scale, scale)
        self._window_size = np.multiply(self._shape, self._resize_scale)
        self._display_surface = pg.display.set_mode(self._window_size, pg.RESIZABLE)

    def handle_pg_events(self, events):
        for event in events:
            match event.type:
                case pg.QUIT:
                    self._is_running = False
                case pg.VIDEORESIZE:
                    self.resize_window((event.w, event.h))

    # def process(self):
    #     if self.food_bitmap is not None:
    #         bg_image = cv2.cvtColor(self.food_bitmap, cv2.COLOR_GRAY2RGB)
    #         bg_surface = pg.surfarray.make_surface(bg_image)
    #         bg_surface = pg.transform.scale(bg_surface, self._window_size)
    #         self._display_surface.blit(bg_surface, (0, 0))

    #     self.pipeline.process_frame(self.current_bitmap)
    #     image = cv2.cvtColor(self.current_bitmap, cv2.COLOR_GRAY2RGB)
    #     surface = pg.surfarray.make_surface(image)
    #     surface.set_colorkey((0, 0, 0))
    #     surface = pg.transform.scale(surface, self._window_size)
    #     self._display_surface.blit(surface, (0, 0))
    #     pg.display.flip()

    def process(self):
        self.pipeline.process_frame(self.current_bitmap)

        if self.food_bitmap is not None:
            image = np.stack([
                self.current_bitmap,
                self.current_bitmap,
                self.food_bitmap,
            ], axis=2)
        else:
            image = cv2.cvtColor(self.current_bitmap, cv2.COLOR_GRAY2RGB)

        surface = pg.surfarray.make_surface(image)
        surface = pg.transform.scale(surface, self._window_size)
        self._display_surface.blit(surface, (0, 0))
        pg.display.flip()

    def run(self):
        self.init_pygame()
        frame_count = 0

        while self._is_running:
            self.handle_pg_events(pg.event.get())
            self.process()

            if frame_count % 30 == 0:
                pg.display.set_caption(f"physarum - FPS: {self._clock.get_fps():.2f}")

            frame_count += 1
            self._clock.tick(FPS_LIMIT)
