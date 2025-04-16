# Physarum

A Python simulation of Physarum Polycephalum (slime mold) behavior using particle-based modeling. This project implements a bio-inspired algorithm that mimics the foraging and network formation patterns of slime mold.

## Description

This simulation uses a particle system to model the behavior of Physarum Polycephalum, where each particle represents a small portion of the organism. The particles deposit and follow pheromone trails, creating emergent patterns similar to those observed in real slime mold colonies. The simulation includes:

- Particle-based movement and trail formation
- Pheromone diffusion and decay
- Sensory-driven particle behavior
- Real-time visualization using Pygame

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) - python package menager
- Required packages:
  - numba==0.61.0
  - numpy==2.1.3
  - opencv-python==4.11.0.86
  - pygame==2.6.1

## Installation

1. Clone the repository:

```zsh
git clone https://github.com/Shikoqu/Physarum.git
cd Physarum/physarum-app
```

aaand... thats it! With **uv** everything is easier :)

## Running the Simulation

To run the simulation, execute the following command from the project root directory:

```zsh
uv run -m main
```

or (if you want to visualize positions of particles and theirs sensors)

```zsh
uv run -m main_with_trace
```

## Project Structure

- `app/main.py` - Main simulation entry point
- `app/engine.py` - Core simulation engine
- `app/particles/` - Particle system implementation
- `app/processing/` - Shaders and image processing
- `app/assets/` - Image resources
- `app/config.py` - Simulation parameters and configuration

## Configuration

You can modify simulation parameters in `app/config.py`, including:

- Number of particles
- Sensor angles and distances
- Diffusion and decay rates
- Movement parameters
