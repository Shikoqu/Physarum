from numba import njit, prange
import numpy as np


def sample_particles_with_trace(
        noise: np.ndarray,
        positions: np.ndarray,
        particles_bitmap: np.ndarray,
) -> np.ndarray:
    return _sample_particles_with_trace(
        noise,
        positions,
        particles_bitmap,
    )

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _sample_particles_with_trace(
        noise: np.ndarray,
        positions: np.ndarray,
        particles_bitmap: np.ndarray,
) -> np.ndarray:
    num_particles = positions.shape[0]
    samples = np.zeros(num_particles, dtype=np.float32)
    particles_bitmap[:] = 0

    for i in prange(num_particles):
        particles_bitmap[
            int(positions[i, 0]),
            int(positions[i, 1])
        ] = 255
        samples[i] = noise[
            int(positions[i, 0]),
            int(positions[i, 1])
        ]

    return samples
