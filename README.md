# Physarum

A Python simulation of Physarum Polycephalum (slime mold) behavior using particle-based modeling. This project implements a bio-inspired algorithm that mimics the foraging and network formation patterns of slime mold.

## Description

This simulation uses a particle system to model the behavior of Physarum Polycephalum, where each particle represents a small portion of the organism. The particles deposit and follow pheromone trails, creating emergent patterns similar to those observed in real slime mold colonies. The simulation includes:

- Particle-based movement and trail formation
- Pheromone diffusion and decay
- Sensory-driven particle behavior
- Real-time visualization using Pygame

## Requirements

- Python 3.x
- Required packages (in `requirements.txt`):
  - numpy==2.1.3
  - opencv-python==4.11.0.86
  - pygame==2.6.1
  - scipy==1.15.2
  - llvmlite==0.44.0
  - numba==0.61.0

## Installation

1. Clone the repository:

```bash
git clone [your-repository-url]
cd Physarum
```

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation, execute the following command from the project root directory:

```bash
python -m app.main
```

or (if you want to visualize positions of particles and theirs sensors)

```bash
python -m app.main_with_trace
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
