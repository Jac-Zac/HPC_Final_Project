import subprocess

import numpy as np
from stencil_reference import inject_energy, total_energy, update_plane


def run_c_version(mode="serial", size=10, iterations=5, periodic=0, seed=1234):
    exe = f"./stencil_{mode}"
    cmd = [
        exe,
        "-x",
        str(size),
        "-y",
        str(size),
        "-n",
        str(iterations),
        "-p",
        str(periodic),
        "-e",
        "1",
        "-E",
        "1.0",
        "-o",
        "1",
    ]
    env = {"SEED": str(seed)}  # if you add seed support
    out = subprocess.check_output(cmd, env={**env, **dict()})
    energies = []
    for line in out.decode().splitlines():
        if "updated system energy" in line:
            energies.append(float(line.split()[-1]))
    return energies


def test_against_reference():
    size, iterations = 10, 5
    periodic = 0
    sources = [(5, 5)]
    grid = np.zeros((size + 2, size + 2))

    # reference Python simulation
    ref_energies = []
    for step in range(iterations):
        inject_energy(periodic, sources, 1.0, grid)
        grid = update_plane(periodic, grid)
        ref_energies.append(total_energy(grid))

    # run C
    c_energies = run_c_version("serial", size, iterations, periodic)

    assert np.allclose(ref_energies, c_energies, rtol=1e-8)
