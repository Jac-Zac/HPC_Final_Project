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
├── python_plotting/       # Performance analysis and plotting scripts
│   ├── plot_strong_scaling.py    # Main plotting script
│   ├── mpi_results.txt          # Sample MPI performance data
│   └── openmp_scaling.txt       # Sample OpenMP performance data
├── slurm_files/           # HPC job scripts for Cineca and Orfeo
├── final_report/          # Project documentation and reports
├── Makefile               # Build system
├── AGENTS.md             # Agent guidelines for coding assistants
└── README.md
```

## Quick Start

### Requirements

**Core Dependencies:**
- GCC 12+ with OpenMP support
- OpenMPI/MPICH 4.1+
- Python 3.7+ (for testing and plotting)

**Python Packages (for plotting and testing):**
```bash
pip install numpy pytest matplotlib pandas
```

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

## Performance Analysis & Plotting

### Data Collection

The simulation outputs performance data in CSV format for analysis:

```bash
# Run tests to generate performance data
make test

# Or run manually with custom parameters
mpirun -np 4 ./stencil_parallel -x 1000 -y 1000 -n 100 -o 1
```

### Scaling Analysis Plots

Use the comprehensive plotting script for performance analysis:

#### Installation
```bash
# Install required Python packages
pip install matplotlib pandas numpy
```

#### Basic Usage
```bash
cd python_plotting/

# Plot with default data files
python plot_strong_scaling.py

# Plot specific data files
python plot_strong_scaling.py mpi_results.txt openmp_scaling.txt

# Save plots without displaying
python plot_strong_scaling.py --no-show --save-dir results/
```

#### Advanced Options

```bash
# Plot only MPI scaling
python plot_strong_scaling.py --mpi-only mpi_data.csv

# Plot only OpenMP scaling
python plot_strong_scaling.py --openmp-only openmp_data.csv

# Custom save directory
python plot_strong_scaling.py --save-dir my_plots/

# Headless mode (no display, just save)
python plot_strong_scaling.py --no-show --save-dir plots/
```

### Generated Plots

The script generates multiple performance analysis plots:

1. **Total Time Scaling** - Overall execution time vs. number of tasks/threads
2. **Computation Time Scaling** - Pure computation time (excludes communication)
3. **Communication Time Scaling** - MPI communication overhead (linear scale)
4. **Energy Computation Time** - Time spent computing energy statistics
5. **Speedup Comparison** - Actual speedup vs. ideal linear speedup
6. **Efficiency Comparison** - Parallel efficiency percentage

### Data Format

Performance data should be in CSV format with these columns:

**MPI Data:**
```csv
Tasks,TotalTime,MaxCompTime,MaxCommTime,EnergyCompTime
1,2.267837,2.255214,0.012534,0.011233
2,1.341955,1.318463,0.030196,0.004693
4,1.146048,1.124610,0.029161,0.004172
```

**OpenMP Data:**
```csv
Threads,TotalTime,MaxCompTime,MaxCommTime,EnergyCompTime
1,2.220450,2.205599,0.014735,0.004812
2,1.387295,1.375457,0.011730,0.004600
4,1.167539,1.156154,0.011290,0.004162
```

### Key Features

- **Automatic Ideal Scaling**: Compares measured performance against theoretical limits
- **MPI vs OpenMP Comparison**: Side-by-side analysis of different parallelization strategies
- **Smart Scaling**: Logarithmic axes for time plots, linear for efficiency
- **Flexible Input**: Handles missing columns gracefully
- **Publication Ready**: High-resolution PNG output (300 DPI)
- **Headless Support**: Perfect for HPC environments without displays

### Interpreting Results

- **Speedup > 1**: Successful parallelization
- **Efficiency > 80%**: Good parallel scaling
- **Communication Time**: Should remain small compared to computation time
- **OpenMP**: Typically better for shared-memory systems
- **MPI**: Better for distributed-memory systems

### Example Output

```
Loading MPI data from mpi_results.txt...
Loading OpenMP data from openmp_scaling.txt...

=== SUMMARY STATISTICS ===
MPI - Max speedup: 15.2x at 64 tasks
MPI - Best efficiency: 78.1% at max scaling
OpenMP - Max speedup: 8.7x at 32 threads
OpenMP - Best efficiency: 85.3% at 32 threads
```

## References

- [OpenMP Documentation](https://www.openmp.org)
- [MPI Standard](https://www.mpi-forum.org)

**Author**: Jacopo Zacchigna - University HPC Final Project
