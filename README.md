# HPC Heat-Stencil Simulation

A 2D heat-stencil simulation with serial and parallel (MPI+OpenMP) implementations for HPC systems.

## Features

- Serial and parallel heat diffusion simulation
- Configurable grid sizes, heat sources, and iterations
- Periodic boundary support
- Energy logging and performance timing
- MPI/OpenMP hybrid parallelism
- Python reference implementation for validation

## Project Structure

```
.
├── include/               # Header files
├── src/                   # Source code (stencil_serial.c, stencil_parallel.c)
├── tests/                 # Python tests and reference implementation
├── python_plotting/       # Scaling analysis scripts
├── slurm_files/           # HPC job scripts for Cineca and Orfeo
├── final_report/          # Project documentation
├── Makefile               # Build system
└── README.md
```

## Quick Start

### Requirements

- GCC 12+ with OpenMP
- OpenMPI/MPICH
- Python 3 (for tests)

### Build

```bash
# Parallel version (default)
make

# Serial version
make MODE=serial

# Clean
make clean
```

### Run Locally

```bash
# Serial (fixed: 100x100 grid, 4 sources, 50 iterations)
./stencil_serial

# Parallel (4 MPI tasks, fixed parameters)
mpirun -np 4 ./stencil_parallel
```

### Fixed Parameters

The implementation uses fixed parameters for simplicity:

- **Grid size**: 100×100
- **Heat sources**: 4 sources at positions [(25,25), (75,25), (25,75), (75,75)]
- **Iterations**: 50
- **Energy per source**: 1.0
- **Boundary conditions**: Non-periodic
- **Output**: Energy statistics and binary dumps at each step

## HPC Execution

### Cineca (Leonardo/Booster)

Use the provided SLURM scripts in `slurm_files/cineca/`:

```bash
# Edit go_dcgp or go_booster with your parameters
sbatch go_dcgp  # For DCGP partition
sbatch go_booster  # For Booster partition
```

Key settings:

- Load modules: `gcc/12.2.0`, `openmpi/4.1.6--gcc--12.2.0`
- Set `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`
- Use `--exclusive` for full node access

### Orfeo (EPYC)

Use scripts in `slurm_files/orfeo/`:

```bash
sbatch mpi_scaling  # MPI scaling test with NUMA awareness
sbatch openmp_scaling  # OpenMP scaling test
```

Features NUMA-aware rankfile generation for optimal performance.

### General HPC Tips

- Pin threads to cores: `export OMP_PLACES=cores`
- Use thread binding: `export OMP_PROC_BIND=close` or `spread`
- Request exclusive nodes for consistent performance
- Monitor affinity: `export OMP_DISPLAY_AFFINITY=TRUE`

## Testing

```bash
# Run all tests (compares serial vs parallel with fixed parameters)
make test

# Python tests only (requires numpy)
source .env/bin/activate && pytest -v tests/

# Manual testing
./stencil_serial          # Serial version
mpirun -np 4 ./stencil_parallel  # Parallel version (4 MPI processes)
```

### Test Output

The test generates binary files for each iteration:

- `plane_XXXXX.bin`: Serial version output
- `plane_global_XXXXX.bin`: Parallel version output

Both versions use identical parameters:

- 100×100 grid
- 4 heat sources at [(25,25), (75,25), (25,75), (75,75)]
- 50 iterations
- Non-periodic boundaries

The parallel version distributes sources across MPI ranks based on domain decomposition.

## Performance Analysis

- Use `python_plotting/plot_strong_scaling.py` for scaling plots
- Enable logging (`LOG=1`) for detailed timing
- Compare serial vs parallel with different thread/MPI configurations

## References

- [OpenMP Documentation](https://www.openmp.org)
- [MPI Standard](https://www.mpi-forum.org)

**Author**: Jacopo Zacchigna - University HPC Final Project
