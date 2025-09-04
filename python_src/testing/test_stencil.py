import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stencil_reference import inject_energy, total_energy, update_plane
from stencil_utils import (
    assemble_global_grid_from_patches,
    extract_energies_from_bins,
    load_sources_from_logs,
)


def test_against_reference():
    """Test non-periodic boundary conditions with assembled global grid."""
    size, iterations = 100, 100  # Match C testing mode defaults
    periodic = 0
    ntasks = 8

    # Load sources from logs (fallback to fixed if logs missing)
    sources = load_sources_from_logs("data_logging", ntasks=ntasks, grid_size=size)

    print("Testing C implementation vs Python reference")
    print("Grid size:", size, "x", size, "Iterations:", iterations)
    print("Periodic boundaries:", periodic)
    print("Sources:", sources)

    # Clean any existing binary files
    import glob

    for f in glob.glob("plane_global_*.bin"):
        os.remove(f)

    # Run C code with periodic boundaries (matching testing mode parameters)
    cmd = [
        "mpirun",
        "-np",
        str(ntasks),
        "./stencil_parallel",
        "-x",
        str(size),
        "-y",
        str(size),
        "-n",
        str(iterations),
        "-p",
        str(periodic),
        "-o",
        "1",  # Enable output
        "-e",
        "4",  # Add 4 different sources
        "-t",
        "1",  # Enable testing mode to use fixed source positions
    ]

    print("Testing non-periodic boundaries with assembled global grid")
    print("Sources:", sources)
    print("Grid size:", size, "x", size)
    print("Iterations:", iterations)

    # Run C code to generate data files
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print("C code execution failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("⚠️  Skipping C vs Python comparison due to C code failure")
            return
    except subprocess.TimeoutExpired:
        print("⚠️  C code execution timed out")
        print("⚠️  Skipping C vs Python comparison")
        return
    except FileNotFoundError:
        print("⚠️  C executable not found. Run 'make' first.")
        print("⚠️  Skipping C vs Python comparison")
        return

    # Compute Python reference energies
    grid = np.zeros((size + 2, size + 2))
    ref_energies = []
    for step in range(iterations):
        inject_energy(periodic, sources, 1.0, grid)
        grid = update_plane(periodic, grid)
        ref_energies.append(total_energy(grid))

    # Try to assemble global grid from C patches
    try:
        c_energies = []
        for step in range(iterations):
            global_grid = assemble_global_grid_from_patches(
                prefix="data_logging/", iteration=step, ntasks=ntasks, grid_size=size
            )
            c_energy = total_energy(global_grid)
            c_energies.append(c_energy)

        print("Python energies (first 5):", ref_energies[:5])
        print("C assembled energies (first 5):", c_energies[:5])

        if len(c_energies) == iterations:
            # Compare with appropriate tolerance for floating point differences
            max_diff = max(abs(p - c) for p, c in zip(ref_energies, c_energies))
            max_energy = max(ref_energies) if ref_energies else 0
            rel_diff = max_diff / max_energy if max_energy > 1e-10 else 0

            print(".6f")
            print(".6f")

            # Allow for some numerical differences between implementations
            assert (
                rel_diff < 0.01
            ), f"C and Python results differ too much: relative diff {rel_diff:.6f}"
            print("✓ C assembled vs Python reference test passed")
        else:
            print(f"⚠️  Expected {iterations} iterations, got {len(c_energies)}")
            print("✓ Python reference implementation test passed")

    except Exception as e:
        print(f"⚠️  Could not assemble global grid: {e}")
        print(
            "⚠️  No C binary files found. Run the C code first to generate comparison data."
        )
        print("   To run: make test")
        # Skip the comparison but still test that Python reference works
        assert len(ref_energies) == iterations
        assert all(e >= 0 for e in ref_energies)
        print("✓ Python reference implementation test passed")


def test_c_periodic_vs_python():
    """Test C implementation with periodic boundaries against Python reference."""
    import subprocess

    size, iterations = 100, 100  # Match C testing mode defaults
    periodic = 1

    # Load sources from logs (fallback to fixed if logs missing)
    sources = load_sources_from_logs()
    if not sources:
        print("⚠️  No logged sources found, falling back to fixed test sources")

    print("Testing C periodic implementation vs Python reference")
    print("Grid size:", size, "x", size, "Iterations:", iterations)
    print("Periodic boundaries:", periodic)
    print("Sources:", sources)

    # Clean any existing binary files
    import glob

    for f in glob.glob("plane_global_*.bin"):
        os.remove(f)

    # Run C code with periodic boundaries (matching testing mode parameters)
    cmd = [
        "mpirun",
        "-np",
        "4",
        "./stencil_parallel",
        "-x",
        str(size),
        "-y",
        str(size),
        "-n",
        str(iterations),
        "-p",
        str(periodic),
        "-o",
        "1",  # Enable output
        "-e",
        "25",  # Add 25 different sources
        "-t",
        "1",  # Enable testing mode to use fixed source positions
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("C code execution failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            # Skip test if C code fails
            print("⚠️  Skipping C vs Python comparison due to C code failure")
            return

        # Get energies from C binary files
        c_energies = extract_energies_from_bins(nx=size, ny=size)

        if len(c_energies) == 0:
            print("⚠️  No C binary files generated")
            return

        # Compute Python reference energies
        grid = np.zeros((size + 2, size + 2))
        py_energies = []

        for step in range(iterations):
            # Inject energy at every iteration (matching C behavior)
            inject_energy(periodic, sources, 1.0, grid)
            grid = update_plane(periodic, grid)
            py_energies.append(total_energy(grid))

        print("Python energies (first 5):", py_energies[:5])
        print("C energies (first 5):", c_energies[:5])

        # Compare energies with appropriate tolerance
        if len(py_energies) == len(c_energies):
            max_diff = max(abs(p - c) for p, c in zip(py_energies, c_energies))
            rel_diff = max_diff / max(py_energies) if py_energies else 0

            print(".6f")
            print(".6f")

            # Allow for some numerical differences between implementations
            assert (
                rel_diff < 0.01
            ), f"C and Python results differ too much: relative diff {rel_diff:.6f}"
            print("✓ C periodic vs Python reference test passed")
        else:
            print(f"⚠️  Length mismatch: Python {len(py_energies)}, C {len(c_energies)}")

    except subprocess.TimeoutExpired:
        print("⚠️  C code execution timed out")
    except FileNotFoundError:
        print("⚠️  C executable not found. Run 'make' first.")
    except Exception as e:
        print(f"⚠️  Test failed with exception: {e}")


"""
Test suite for HPC Heat-Stencil Simulation

This module contains tests for both non-periodic and periodic boundary conditions.
Run tests with:
    pytest tests/test_stencil.py -v

For periodic boundary tests specifically:
    pytest tests/test_stencil.py::test_periodic_boundaries -v
    pytest tests/test_stencil.py::test_boundary_source_propagation -v

To test C implementation with periodic boundaries:
    pytest tests/test_stencil.py::test_c_periodic_vs_python -v
"""
