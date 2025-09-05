# HPC Heat-Stencil Simulation

A 2D heat-stencil simulation with serial and parallel (MPI+OpenMP) implementations for HPC systems.

## Features

- Serial and parallel heat diffusion simulation
- Configurable grid sizes, heat sources, and iterations
- Periodic boundary support
- Energy logging and performance timing
- MPI/OpenMP hybrid parallelism
- Python reference implementation for validation
- Source location logging with testing option (-t)
- Shared utilities for grid assembly and visualization
- Interactive visualization of simulation results

## Project Structure

```
.
├── include/               # Header files
├── src/                   # Source code (stencil_serial.c, stencil_parallel.c)
├── tests/                 # Python tests and reference implementation
│   ├── test_stencil.py    # Comprehensive test suite
│   └── stencil_reference.py # Python reference implementation
├── python_plotting/       # Performance analysis and plotting scripts
│   ├── generate_visualizations.py # Visualization generation script
│   ├── plot_strong_scaling.py     # Main plotting script
│   └── stencil_utils.py           # Shared utilities for grid assembly & viz
├── slurm_files/           # HPC job scripts for Cineca and Orfeo
├── results/               # Output results directory
├── CODE_REVIEW.md         # Comprehensive code review and analysis
├── Makefile               # Build system
└── README.md
```

## Quick Start

### Requirements

**Core Dependencies:**

- GCC with OpenMP support
- OpenMPI/MPICH 4.1+
- Python 3.7+ (for testing and plotting)

**Python Setup (Recommended):**

```bash
# Create virtual environment
python3 -m venv .env
source .env/bin/activate

# Install required packages
pip install numpy pytest matplotlib pandas
```

**Python Packages (for plotting and testing):**

- numpy: Array operations and numerical computing
- pytest: Testing framework
- matplotlib: Plotting and visualization
- pandas: Data manipulation and CSV handling

**Optional (for development):**

- clangd (for LSP support)
- valgrind (for memory debugging)

### Build

```bash
# Parallel version (default)
make

# Serial version
make MODE=serial

# Clean
make clean
```

**Note:** The serial version uses fixed parameters and does not support command line options.

### Run Locally

```bash
# Serial (fixed: 100x100 grid, 4 sources, 50 iterations)
./stencil_serial

# Parallel (4 MPI tasks, default parameters: 10000x10000 grid, 1000 iterations)
mpirun -np 4 ./stencil_parallel

# Parallel with source location logging
mpirun -np 4 ./stencil_parallel -t 1

# Custom parameters
mpirun -np 4 ./stencil_parallel -x 200 -y 200 -n 100 -e 8 -p 1
```

### Command Line Options

The parallel version supports extensive customization:

```bash
./stencil_parallel [options]

Options:
  -x <size>    Grid width (default: 10000)
  -y <size>    Grid height (default: 10000)
  -n <iter>    Number of iterations (default: 1000)
  -e <num>     Number of heat sources (default: 4)
  -E <energy>  Energy per source (default: 1.0)
  -p <0|1>     Periodic boundaries (0=no, 1=yes)
  -o <0|1>     Output energy stats (0=no, 1=yes)
  -t <0|1>     Testing mode - save source locations
  -v <level>   Verbosity level
  -h           Show help
```

### Fixed Parameters

**Serial version** uses fixed parameters:

- **Grid size**: 100×100
- **Heat sources**: 4 sources at positions [(25,25), (75,25), (25,75), (75,75)]
- **Iterations**: 50
- **Energy per source**: 1.0
- **Boundary conditions**: Non-periodic
- **Output**: Energy statistics and binary dumps at each step

**Parallel version** has configurable parameters with defaults:

- **Grid size**: 10000×10000
- **Heat sources**: 4 sources (positions depend on grid decomposition)
- **Iterations**: 1000
- **Energy per source**: 1.0
- **Boundary conditions**: Non-periodic
- **Output**: Energy statistics (binary dumps when -o 1)

## HPC Execution

### Cineca (Leonardo/Booster)

Use the provided SLURM scripts in `slurm_files/cineca/`:

```bash
# Edit go_dcgp or go_booster with your parameters
sbatch go_dcgp  # For DCGP partition
sbatch go_booster  # For Booster partition
```

Key settings:

- Load appropriate GCC and OpenMPI modules
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

### Quick Test Commands

```bash
# Run all tests (compares serial vs parallel with fixed parameters)
make test

# Python tests only (activate virtual environment first)
source .env/bin/activate && pytest -v tests/

# Manual testing
./stencil_serial          # Serial version
mpirun -np 4 ./stencil_parallel  # Parallel version (4 MPI processes)

# Testing with source location logging
mpirun -np 4 ./stencil_parallel -t 1
```

### Test Output Files

The test generates several types of output files:

**Standard Output:**

- `plane_XXXXX.bin`: Serial version output per iteration
- `plane_global_XXXXX.bin`: Parallel version assembled output

**Testing Mode Output:**

- `data_logging/sources_rank*.txt`: Source locations for each MPI rank
- `data_logging/X_plane_XXXXX.bin`: Per-rank binary data for parallel validation

### Advanced Testing Features

The testing framework includes:

- **Source validation**: Compares source locations between C and Python implementations
- **Grid assembly**: Reconstructs full simulation grid from distributed MPI patches
- **Energy conservation**: Validates that total energy is conserved across iterations
- **Parallel correctness**: Ensures MPI decomposition produces correct results

### Test Output Files

The test generates several types of output files:

## Performance Analysis & Plotting

### Scaling Analysis Plots

Use the comprehensive plotting script for performance analysis:

#### Basic Usage

```bash
cd python_plotting/

# Plot with data files (provide your own CSV files)
python plot_strong_scaling.py data_file.csv

# Save plots without displaying
python plot_strong_scaling.py data_file.csv --no-show --save-dir results/
```

#### Advanced Options

```bash
# Plot only MPI scaling
python plot_strong_scaling.py --mpi-only data.csv

# Plot only OpenMP scaling
python plot_strong_scaling.py --openmp-only data.csv

# Custom save directory
python plot_strong_scaling.py data.csv --save-dir my_plots/

# Headless mode (no display, just save)
python plot_strong_scaling.py data.csv --no-show --save-dir plots/
```

### Generated Plots

The script generates multiple performance analysis plots:

1. **Total Time Scaling** - Overall execution time vs. number of tasks/threads
2. **Computation Time Scaling** - Pure computation time (excludes communication)
3. **Communication Time Scaling** - MPI communication overhead (linear scale)
4. **Energy Computation Time** - Time spent computing energy statistics
5. **Speedup Comparison** - Actual speedup vs. ideal linear speedup
6. **Efficiency Comparison** - Parallel efficiency percentage

## Simulation Visualization

The plotting utilities include visualization capabilities for analyzing simulation results.

### Prerequisites

```bash
make visualization
```

### Visualization Features

- **Heat Map Display**: Color-coded energy distribution across the grid
- **Source Location Markers**: Visual indicators for heat source positions
- **Grid Assembly**: Automatically reconstructs full simulation from MPI patches
- **Interactive Plots**: Zoom, pan, and save capabilities
- **Multiple Formats**: PNG, PDF, SVG export options

This creates a time series of the heat diffusion process with source locations clearly marked.

## References

- [OpenMP Documentation](https://www.openmp.org)
- [MPI Standard](https://www.mpi-forum.org)

## Code Quality

This project has undergone comprehensive code review and analysis. See [`CODE_REVIEW.md`](CODE_REVIEW.md) for detailed findings, performance analysis, and recommendations.

**Overall Assessment: A- (90/100)**
- Professional project structure with excellent MPI+OpenMP hybrid parallelization
- Comprehensive testing and validation framework
- Ready for production HPC environments
- Minor code quality improvements have been applied

**Author**: Jacopo Zacchigna - University HPC Final Project
**Last Updated**: September 2025
