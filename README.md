# HPC Heat-Stencil Simulation

## Overview

This project implements a **2D heat-stencil simulation** with both **serial** and **parallel** versions.
The parallel version uses **MPI** and **OpenMP**. The simulation supports multiple heat sources, configurable grid sizes, iterations, optional periodic boundaries, and energy logging.

---

## Features

- Serial and parallel implementations
- Configurable grid size (`x` × `y`)
- Multiple heat sources and energy per source
- Adjustable iterations
- Periodic boundary support
- Logging of energy budget
- MPI/OpenMP hybrid parallelism

---

## Project Structure

```bash
.
├── include/               # Header files
│   └── stencil.h
├── src/                   # Source files
│   ├── stencil_template_serial.c
│   └── stencil_template_parallel.c
├── Makefile               # Build instructions
└── README.md
```

---

## Compilation

### Requirements

- `gcc-15` or later
- `mpicc` (Open MPI or MPICH)
- OpenMP support

### Build

**Serial:**

```bash
make MODE=serial
```

**Parallel (MPI/OpenMP):**

```bash
make MODE=parallel
```

**Enable logging:**

```bash
make MODE=parallel LOG=1
```

**Clean:**

```bash
make clean
```

---

## Command-Line Options

| Option | Description                   | Default |
| ------ | ----------------------------- | ------- |
| `-x`   | Grid width                    | 1000    |
| `-y`   | Grid height                   | 1000    |
| `-e`   | Number of heat sources        | 1       |
| `-E`   | Energy per source             | 1.0     |
| `-n`   | Number of iterations          | 99      |
| `-p`   | Periodic boundaries (0/1)     | 0       |
| `-o`   | Print energy per step (0/1)   | 0       |
| `-f`   | Frequency of energy injection | 0.0     |
| `-h`   | Show help message             | N/A     |

**Example:**

```bash
./stencil_template_serial -x 500 -y 500 -n 50 -e 2 -E 2.0 -o 1
```

---

## Running Parallel Version

```bash
mpirun -np 4 ./stencil_template_parallel -x 1000 -y 1000 -n 100 -e 4 -E 1.5 -o 1
export OMP_NUM_THREADS=4
```

---

## Timing & Logging

- Compile with `LOG=1` for per-iteration logging.
- Measure total execution:

```bash
time ./stencil_template_parallel -x 1000 -y 1000 -n 100
```

- MPI timing inside code: use `MPI_Wtime()`.

---

## Suggested Test Cases

| Grid Size | Sources | Energy | Iterations | MPI / OMP |
| --------- | ------- | ------ | ---------- | --------- |
| 500×500   | 1       | 1.0    | 50         | 2         |
| 1000×1000 | 2       | 2.0    | 100        | 4         |
| 2000×2000 | 4       | 1.5    | 200        | 8         |

---

## Notes

- Compiler should support **C17** or **C99+**.
- Periodic boundaries wrap heat propagation at edges.

---

## References

- [OpenMP](https://www.openmp.org)
- [MPI](https://www.mpi-forum.org)

---

## Author

- University HPC Project by Jacopo Zacchigna
