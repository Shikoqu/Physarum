import cv2
import numpy as np
import pygame as pg
from datetime import datetime
import os

from app.config import FPS_LIMIT
from app.processing.pipeline import Pipeline


class Engine:
    def __init__(
            self,
            pipeline: Pipeline,
            bitmap: np.ndarray,
            food_bitmap: np.ndarray = None,
    ):
        self.pheromone_bitmap: np.ndarray = bitmap
        self.pipeline: Pipeline = pipeline
        self.food_bitmap: np.ndarray = food_bitmap

        self._clock: pg.time.Clock | None = None
        self._is_running: bool = False
        self._is_stopped: bool = True
        self._shape: tuple[int, int] = bitmap.shape
        self._resize_scale: tuple[float, float] = (1., 1.)
        self._window_size: tuple[int, int] | None = None
        self._display_surface: pg.Surface | None = None
        self._frame_count: int = 0
        self._save_path: str = f"saves/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self._directory_exists: bool = False

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

    def make_image(self):
        # rgb = [
        #     self.pheromone_bitmap,
        #     self.pheromone_bitmap,
        #     self.pheromone_bitmap,
        # ]
        rgb = [
            self.food_bitmap,
            self.food_bitmap,
            self.pheromone_bitmap,
        ]
        return np.stack(rgb, axis=2)

    def save_frame(self):
        if not self._directory_exists:
            os.makedirs(self._save_path, exist_ok=True)
            self._directory_exists = True

        filename = f"{self._save_path}/{self._frame_count:06d}.png"
        cv2.imwrite(
            filename, cv2.cvtColor(self.make_image().swapaxes(0, 1), cv2.COLOR_BGR2RGB)
        )

    def handle_pg_events(self, events):
        for event in events:
            match event.type:
                case pg.QUIT:
                    self._is_running = False
                case pg.VIDEORESIZE:
                    self.resize_window((event.w, event.h))
                case pg.KEYDOWN if event.key == pg.K_SPACE:
                    self._is_stopped = not self._is_stopped
                case pg.KEYDOWN if event.key == pg.K_s:
                    self.save_frame()

    def process(self):
        self.pipeline.process_frame(self.pheromone_bitmap)

        surface = pg.surfarray.make_surface(self.make_image())
        surface = pg.transform.scale(surface, self._window_size)
        self._display_surface.blit(surface, (0, 0))
        pg.display.flip()

    def run(self):
        self.init_pygame()
        self.process()

        while self._is_running:
            self.handle_pg_events(pg.event.get())

            if not self._is_stopped:
                self.process()
                self._frame_count += 1

            if self._frame_count % 30 == 0:
                pg.display.set_caption(f"physarum - FPS: {self._clock.get_fps():.2f}")

            self._clock.tick(FPS_LIMIT)
