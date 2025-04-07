from app.particles.particle_system import ParticleSystem


if __name__ == "__main__":
    system = ParticleSystem((1, 1))
    print(system.angles)
    system.get_sensor_positions()
