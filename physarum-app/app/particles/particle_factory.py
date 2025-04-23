import numpy as np
from typing import Callable

from app.config import (
    NUM_PARTICLES,
    PARTICLE_DISTRIBUTION_METHOD,
    PARTICLE_DISTRIBUTION_POINT,
    PARTICLE_DISTRIBUTION_FACE_BACK,
    PARTICLE_DISTRIBUTION_RADIUS,
)


class ParticleFactory:
    def __init__(self):
        self._distribution_methods: dict[str, Callable] = {
            "random": self._random_distribution,
            "center": self._center_distribution,
            "circle": self._circle_distribution,
            "ring": self._ring_distribution,
            "grid": self._grid_distribution,
            "food_based": self._food_based_distribution,
        }
        self._method = self._distribution_methods[PARTICLE_DISTRIBUTION_METHOD.lower()]

    def prime_particles(
        self,
        food_bitmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if PARTICLE_DISTRIBUTION_POINT is not None:
            point = food_bitmap.shape * np.array(PARTICLE_DISTRIBUTION_POINT)
        else:
            point = None

        return self._method(food_bitmap, point, PARTICLE_DISTRIBUTION_FACE_BACK)

    def _random_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = food_bitmap.shape
        positions = np.random.uniform(0, shape, (NUM_PARTICLES, 2))
        angles = self._calculate_angles(positions, point, face_back)
        return positions, angles

    def _center_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.array([point] * NUM_PARTICLES)
        angles = self._calculate_angles(positions, None, face_back)
        return positions, angles

    def _ring_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = PARTICLE_DISTRIBUTION_RADIUS * np.min(food_bitmap.shape)
        angles = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
        positions = np.column_stack((np.cos(angles), np.sin(angles))) * radius + point
        angles = self._calculate_angles(positions, point, face_back)
        return positions, angles

    def _circle_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = PARTICLE_DISTRIBUTION_RADIUS * np.min(food_bitmap.shape)
        angles = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
        random_radius = np.random.uniform(0, radius, NUM_PARTICLES)
        positions = np.column_stack((np.cos(angles), np.sin(angles))) * random_radius[:, np.newaxis] + point
        angles = self._calculate_angles(positions, point, face_back)
        return positions, angles

    def _grid_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = np.array(food_bitmap.shape)
        grid_size = int(np.sqrt(NUM_PARTICLES))

        # Create grid points
        x = np.linspace(0, shape[0], grid_size)
        y = np.linspace(0, shape[1], grid_size)
        xx, yy = np.meshgrid(x, y)

        # Flatten and take first NUM_PARTICLES points
        positions = np.column_stack((xx.ravel(), yy.ravel()))[:NUM_PARTICLES]
        angles = self._calculate_angles(positions, point, face_back)
        return positions, angles

    def _food_based_distribution(
        self,
        food_bitmap: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Normalize food bitmap to get probability distribution
        prob = food_bitmap.astype(float) / food_bitmap.max()
        prob = prob / prob.sum()

        # Flatten and sample positions based on food intensity
        flat_idx = np.random.choice(
            np.prod(food_bitmap.shape),
            size=NUM_PARTICLES,
            p=prob.ravel()
        )
        positions = np.column_stack(np.unravel_index(flat_idx, food_bitmap.shape)).astype(np.float64)
        angles = self._calculate_angles(positions, point, face_back)
        return positions, angles

    def _calculate_angles(
        self,
        positions: np.ndarray,
        point: tuple[int, int] | None,
        face_back: bool,
    ) -> np.ndarray:
        if point is None:
            return np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)

        point = np.array(point)
        dx = point[0] - positions[:, 0]
        dy = point[1] - positions[:, 1]
        angles = np.arctan2(dx, dy)

        if face_back:
            angles = (angles + np.pi) % (2 * np.pi)

        return angles
