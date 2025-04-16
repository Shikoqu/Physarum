from numba import njit, prange
import numpy as np

from app.utils.time_it import time_it, time_it_configure


def sample_particles(
        noise: np.ndarray,
        positions: np.ndarray,
) -> np.ndarray:
    args = (
        noise,
        positions,
    )
    return sample_particles1(*args)
    # return sample_particles2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

# time_it_configure(__name__)

@time_it
def sample_particles1(*args):
    """1M particles: ~0.5 ms"""
    return _sample_particles1(*args)

@time_it
def sample_particles2(*args):
    """1M particles: ~5 ms"""
    return _sample_particles2(*args)

# # # # # # # # # # # # # # # # # # # # # # # # #

@njit(parallel=True)
def _sample_particles1(
        noise: np.ndarray,
        positions: np.ndarray,
) -> np.ndarray:
    num_particles = positions.shape[0]
    samples = np.zeros(num_particles, dtype=np.float32)

    for i in prange(num_particles):
        samples[i] = noise[
            int(positions[i, 0]),
            int(positions[i, 1])
        ]

    return samples

def _sample_particles2(
        noise: np.ndarray,
        positions: np.ndarray,
) -> np.ndarray:
    int_pos = positions.astype(np.uint32)
    return noise[int_pos[:, 0], int_pos[:, 1]]
