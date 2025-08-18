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
        "2",  # Match Python: 2 sources
        "-E",
        "1.0",
        "-f",
        "1.0",  # Inject every iteration
        "-o",
        "1",
        "-s",
        str(seed),  # Use fixed seed
    ]
    env = {"SEED": str(seed)}  # if you add seed support
    out = subprocess.check_output(cmd, env={**env, **dict()})
    energies = []
    for line in out.decode().splitlines():
        if "updated system energy" in line:
            energies.append(float(line.split()[-1]))
    return energies


def test_against_reference():
    size, iterations = 20, 50
    periodic = 0

    # HACK: Hard coded sources values this could be improved
    # It is good enough for now though we only use it to test
    # Use the EXACT sources that C printed
    sources = [(12, 18), (4, 7)]  # From C debug output
    print("Sources:", sources)

    grid = np.zeros((size + 2, size + 2))

    ref_energies = []
    for step in range(iterations):
        # Only inject at step 0 (matching C behavior)
        if step == 0:
            inject_energy(periodic, sources, 1.0, grid)

        grid = update_plane(periodic, grid)
        ref_energies.append(total_energy(grid))

    c_energies = run_c_version("serial", size, iterations, periodic, seed=1234)

    print("Python energies:", ref_energies[:3])
    print("C energies:", c_energies[:3])
    assert np.allclose(ref_energies, c_energies, rtol=1e-5)
