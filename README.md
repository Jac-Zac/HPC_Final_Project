# HPC Heat-Stencil Simulation

A 2D heat-stencil simulation with serial and parallel (MPI+OpenMP) implementations for HPC systems.

## Features

- Serial and parallel heat diffusion simulation
- Configurable grid sizes, heat sources, and iterations
- Periodic boundary support
- Energy logging and performance timing
- MPI/OpenMP hybrid parallelism
- Python reference implementation for validation
- **NEW**: Source location logging with testing option (-t)
- **NEW**: Shared utilities for grid assembly and visualization
- **NEW**: Interactive visualization of simulation results

## Project Structure

```
.
├── include/               # Header files
├── src/                   # Source code (stencil_serial.c, stencil_parallel.c)
├── tests/                 # Python tests and reference implementation
│   ├── test_stencil.py    # Comprehensive test suite
│   └── stencil_reference.py # Python reference implementation
├── python_plotting/       # Performance analysis and plotting scripts
│   ├── plot_strong_scaling.py    # Main plotting script with visualization
│   ├── stencil_utils.py          # Shared utilities for grid assembly & viz
│   ├── mpi_results.txt          # Sample MPI performance data
│   └── openmp_scaling.txt       # Sample OpenMP performance data
├── slurm_files/           # HPC job scripts for Cineca and Orfeo
├── final_report/          # Project documentation and reports
├── .env/                  # Python virtual environment (created by setup)
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

### Run Locally

```bash
# Serial (fixed: 100x100 grid, 4 sources, 50 iterations)
./stencil_serial

# Parallel (4 MPI tasks, fixed parameters)
mpirun -np 4 ./stencil_parallel

# Parallel with source location logging (NEW!)
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
  -t <0|1>     Testing mode - save source locations (NEW!)
  -v <level>   Verbosity level
  -h           Show help
```

### Fixed Parameters (Default)

The implementation uses fixed parameters for simplicity:

- **Grid size**: 100×100
- **Heat sources**: 4 sources at positions [(25,25), (75,25), (25,75), (75,75)]
- **Iterations**: 50
- **Energy per source**: 1.0
- **Boundary conditions**: Non-periodic
- **Output**: Energy statistics and binary dumps at each step

### Testing Mode Features (NEW!)

When using the `-t 1` option, the simulation saves additional data:

- **Source locations**: Saved to `data_logging/sources_rank*.txt` files
- **Per-rank data**: Binary dumps for each MPI rank (`data_logging/X_plane_XXXXX.bin`)
- **Global assembly**: Python utilities can reconstruct the full simulation grid

This enables detailed validation and visualization of the parallel computation.

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

### Quick Test Commands

```bash
# Run all tests (compares serial vs parallel with fixed parameters)
make test

# Python tests only (activate virtual environment first)
source .env/bin/activate && pytest -v tests/

# Manual testing
./stencil_serial          # Serial version
mpirun -np 4 ./stencil_parallel  # Parallel version (4 MPI processes)

# Testing with source location logging (NEW!)
mpirun -np 4 ./stencil_parallel -t 1
```

### Test Output Files

The test generates several types of output files:

**Standard Output:**
- `plane_XXXXX.bin`: Serial version output per iteration
- `plane_global_XXXXX.bin`: Parallel version assembled output

**Testing Mode Output (NEW!):**
- `data_logging/sources_rank*.txt`: Source locations for each MPI rank
- `data_logging/X_plane_XXXXX.bin`: Per-rank binary data for parallel validation

### Test Parameters

Both versions use identical parameters for comparison:

- **Grid size**: 100×100
- **Heat sources**: 4 sources at [(25,25), (75,25), (25,75), (75,75)]
- **Iterations**: 50
- **Boundary conditions**: Non-periodic

### Advanced Testing Features

The testing framework now includes:

- **Source validation**: Compares source locations between C and Python implementations
- **Grid assembly**: Reconstructs full simulation grid from distributed MPI patches
- **Energy conservation**: Validates that total energy is conserved across iterations
- **Parallel correctness**: Ensures MPI decomposition produces correct results

**Example test output:**
```bash
$ pytest tests/test_stencil.py::test_against_reference -v
======================== test session starts ==============================
tests/test_stencil.py::test_against_reference PASSED
Python energies (first 5): [2.0, 4.0, 6.0, 8.0, 10.0]
C assembled energies (first 5): [2.0, 4.0, 6.0, 8.0, 10.0]
Max difference: 0.000000 (0.000000%)
✓ C assembled vs Python reference test passed
```

## Performance Analysis & Plotting

### Data Collection

The simulation outputs performance data in CSV format for analysis:

```bash
# Run tests to generate performance data
make test

# Or run manually with custom parameters
mpirun -np 4 ./stencil_parallel -x 1000 -y 1000 -n 100 -o 1

# Generate data with source logging for visualization
mpirun -np 4 ./stencil_parallel -x 500 -y 500 -n 50 -o 1 -t 1
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

## Simulation Visualization (NEW!)

The enhanced plotting script now includes interactive visualization capabilities for analyzing simulation results.

### Prerequisites

```bash
# Activate virtual environment
source .env/bin/activate

# Install visualization dependencies
pip install matplotlib numpy
```

### Basic Visualization

```bash
cd python_plotting/

# Visualize a specific iteration
python3 -c "
from plot_strong_scaling import visualize_iteration
grid = visualize_iteration(iteration=10, ntasks=4, grid_size=100)
"

# Load and visualize from saved data
python3 -c "
from python_plotting.stencil_utils import load_and_visualize_iteration
grid = load_and_visualize_iteration(iteration=5, ntasks=2, grid_size=50)
"
```

### Visualization Features

- **Heat Map Display**: Color-coded energy distribution across the grid
- **Source Location Markers**: Visual indicators for heat source positions
- **Grid Assembly**: Automatically reconstructs full simulation from MPI patches
- **Interactive Plots**: Zoom, pan, and save capabilities
- **Multiple Formats**: PNG, PDF, SVG export options

### Advanced Visualization

```python
from python_plotting.stencil_utils import (
    load_sources_from_logs,
    assemble_global_grid_from_patches,
    visualize_grid
)

# Load source locations
sources = load_sources_from_logs()
print(f"Found {len(sources)} energy sources")

# Assemble global grid from MPI patches
grid = assemble_global_grid_from_patches(
    prefix='data_logging/',
    iteration=25,
    ntasks=4,
    grid_size=200
)

# Create custom visualization
visualize_grid(
    grid,
    sources=sources,
    title="Heat Stencil - Iteration 25",
    save_path="simulation_snapshot.png"
)
```

### Visualization Output

The visualization shows:
- **Energy Distribution**: Heat map with color scale
- **Source Locations**: Blue circles marking energy injection points
- **Grid Boundaries**: Clear indication of computational domain
- **Scale Information**: Energy values and spatial coordinates

**Example visualization:**
- Grid size: 200×200 (with halos: 202×202)
- Energy range: 0.0 - 2.5 (normalized scale)
- Sources: Marked with blue indicators
- Output: High-resolution PNG/PDF files

### Integration with Testing

The visualization system integrates seamlessly with the testing framework:

```bash
# Run simulation with logging
mpirun -np 4 ./stencil_parallel -x 100 -y 100 -n 50 -t 1 -o 1

# Automatically visualize results
cd python_plotting/
python3 -c "
from plot_strong_scaling import visualize_iteration
for i in [10, 25, 40]:
    visualize_iteration(i, ntasks=4, grid_size=100, save_path=f'iter_{i:03d}.png')
"
```

This creates a time series of the heat diffusion process with source locations clearly marked.

## References

- [OpenMP Documentation](https://www.openmp.org)
- [MPI Standard](https://www.mpi-forum.org)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Recent Updates

### Version 2.0 Features (Latest)

- **🔥 Source Location Logging**: New `-t` testing option saves source positions for validation
- **📊 Shared Utilities Module**: Modular `stencil_utils.py` for reusable grid operations
- **🎨 Interactive Visualization**: Enhanced plotting with heat maps and source markers
- **🧪 Advanced Testing**: Comprehensive test suite with grid assembly validation
- **🔧 Improved Modularity**: Clean separation of concerns between testing and visualization
- **📈 Performance Analysis**: Enhanced scaling plots with efficiency metrics

### Key Improvements

1. **Testing Mode**: Use `-t 1` to save source locations and enable detailed validation
2. **Visualization**: Interactive plots showing energy distribution and source locations
3. **Grid Assembly**: Automatic reconstruction of full simulation from MPI patches
4. **Error Handling**: Robust error recovery for missing files and execution failures
5. **Virtual Environment**: Recommended Python setup with `.env` virtual environment

**Author**: Jacopo Zacchigna - University HPC Final Project
**Last Updated**: September 2025
