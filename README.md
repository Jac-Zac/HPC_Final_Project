# HPC Heat-Stencil Simulation

## Overview

This project implements a **2D heat-stencil simulation** to study heat diffusion on a rectangular plate. It has both **serial** and **parallel** versions. The parallel version uses **MPI** for distributed-memory parallelism and **OpenMP** for shared-memory threading. This project is suitable for HPC assignments, performance studies, and energy propagation experiments.

The simulation supports multiple heat sources, adjustable grid sizes, configurable iterations, and optional periodic boundary conditions. You can log energy values and execution times for analysis.

---

## Features

- Serial and parallel implementations
- Configurable grid size (`x` × `y`)
- Configurable number of heat sources
- Adjustable energy per source
- Adjustable number of iterations
- Periodic boundary support
- Logging of energy budget per iteration
- MPI/OpenMP hybrid parallelism

---

## Project Structure

```
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

The project supports **serial** and **parallel** builds.

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

**Clean built executables:**

```bash
make clean
```

---

## Command-Line Options

The `initialize` function parses the following command-line arguments:

| Option | Description                                             | Default |
| ------ | ------------------------------------------------------- | ------- |
| `-x`   | Grid width                                              | 1000    |
| `-y`   | Grid height                                             | 1000    |
| `-e`   | Number of heat sources                                  | 1       |
| `-E`   | Energy per source                                       | 1.0     |
| `-n`   | Number of iterations                                    | 99      |
| `-p`   | Enable periodic boundaries (0 = false, 1 = true)        | 0       |
| `-o`   | Print energy budget at every step (0 = false, 1 = true) | 0       |
| `-f`   | Frequency of energy injection (0.0–1.0)                 | 0.0     |
| `-h`   | Show help message                                       | N/A     |

Example:

```bash
./stencil_template_serial -x 500 -y 500 -n 50 -e 2 -E 2.0 -o 1
```

---

## Running Parallel Version

Use `mpirun` or `mpiexec`:

```bash
mpirun -np 4 ./stencil_template_parallel -x 1000 -y 1000 -n 100 -e 4 -E 1.5 -o 1
```

- `-np 4` → run 4 MPI processes
- OpenMP threads are determined by `OMP_NUM_THREADS`:

```bash
export OMP_NUM_THREADS=4
```

---

## Logging and Timing

### Option 1: Built-in logging

Compile with `LOG=1`:

```bash
make MODE=parallel LOG=1
```

- Prints per-iteration energy and timing info (if implemented in `stencil_template_parallel.c`).

### Option 2: Measure total execution

```bash
time ./stencil_template_parallel -x 1000 -y 1000 -n 100
```

- Reports real, user, and system time.

### Option 3: MPI timing (inside code)

The parallel version may use `MPI_Wtime()`:

```c
double start = MPI_Wtime();
// run stencil
double end = MPI_Wtime();
printf("Elapsed time: %f seconds\n", end-start);
```

---

## Suggested Test Cases

| Grid Size | Sources | Energy | Iterations | MPI / OMP |
| --------- | ------- | ------ | ---------- | --------- |
| 500×500   | 1       | 1.0    | 50         | 2         |
| 1000×1000 | 2       | 2.0    | 100        | 4         |
| 2000×2000 | 4       | 1.5    | 200        | 8         |

- Small grids for debugging
- Large grids for scalability testing

---

## Notes

- Ensure your compiler supports **C17** or **C99+** for the `restrict` keyword.
- Header files are treated as C (`-x c`) to avoid LSP issues.
- Periodic boundary conditions wrap heat propagation at edges when enabled.

---

## References

- OpenMP Documentation: [https://www.openmp.org](https://www.openmp.org)
- MPI Documentation: [https://www.mpi-forum.org](https://www.mpi-forum.org)
- HPC stencil computation literature (heat diffusion, finite differences)

---

## Author

- University HPC Project by Jacopo Zacchigna
