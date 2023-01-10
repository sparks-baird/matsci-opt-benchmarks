from os import path

from matsci_opt_benchmarks.particle_packing.utils.packing_generation import (
    particle_packing_simulation,
)


def test_particle_packing_simulation():
    results = particle_packing_simulation(
        [0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3],
        [0.25, 0.25, 0.5],
        util_dir=path.join("src", "matsci_opt_benchmarks", "particle_packing", "utils"),
        data_dir=path.join("data", "interim", "particle_packing"),
    )
    print(results)


if __name__ == "__main__":
    test_particle_packing_simulation()


1 + 1
