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

# With logging
make LOG=1

# Clean
make clean
```

### Run Locally

```bash
# Serial
./stencil_serial -x 1000 -y 1000 -n 100 -e 4 -E 1.5

# Parallel (4 MPI tasks, 4 OpenMP threads each)
export OMP_NUM_THREADS=4
mpirun -np 4 ./stencil_parallel -x 1000 -y 1000 -n 100 -e 4 -E 1.5
```

### Command Line Options

- `-x, -y`: Grid dimensions (default: 1000x1000)
- `-n`: Iterations (default: 99)
- `-e`: Number of heat sources (default: 1)
- `-E`: Energy per source (default: 1.0)
- `-p`: Periodic boundaries (0/1, default: 0)
- `-o`: Print energy per step (0/1, default: 0)
- `-f`: Energy injection frequency (default: 0.0)

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
# Run all tests
make test

# Python tests only
pytest -v tests/

# Validate against reference implementation
python -m pytest tests/test_stencil.py::test_against_reference
```

## Performance Analysis

- Use `python_plotting/plot_strong_scaling.py` for scaling plots
- Enable logging (`LOG=1`) for detailed timing
- Compare serial vs parallel with different thread/MPI configurations

## References

- [OpenMP Documentation](https://www.openmp.org)
- [MPI Standard](https://www.mpi-forum.org)

**Author**: Jacopo Zacchigna - University HPC Final Project
