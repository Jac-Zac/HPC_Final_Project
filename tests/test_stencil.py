import glob
import re

import numpy as np
from stencil_reference import inject_energy, total_energy, update_plane


def read_global_bin(filename, nx, ny):
    """Read the global binary produced by your MPI C code."""
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape((ny, nx))


def extract_energies_from_bins(prefix="plane_global_", nx=100, ny=100):
    """Read all iteration files and return total energies like C would print."""
    files = sorted(
        glob.glob(f"{prefix}*.bin"), key=lambda x: int(re.search(r"(\d+)", x).group(1))
    )
    energies = []
    for f in files:
        grid = read_global_bin(f, nx, ny)
        # sum interior only, matching C get_total_energy
        energy = np.sum(grid[1:-1, 1:-1])
        energies.append(energy)
    return energies


def test_against_reference():
    size, iterations = 100, 50
    periodic = 0

    # HACK: Hard coded sources values this could be improved
    sources = [(50, 4), (7, 93), (15, 86), (43, 99)]  # your reference

    print("Sources:", sources)

    # Compute Python reference energies
    grid = np.zeros((size + 2, size + 2))
    ref_energies = []
    for step in range(iterations):
        inject_energy(periodic, sources, 1.0, grid)
        grid = update_plane(periodic, grid)
        ref_energies.append(total_energy(grid))

    # Get energies from global binary files
    c_energies = extract_energies_from_bins(nx=size, ny=size)

    print("Python energies:", ref_energies[:3])
    print("C energies:", c_energies[:3])
    assert np.allclose(ref_energies, c_energies, rtol=1e-3)
