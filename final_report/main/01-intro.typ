#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Introduction

== Objective

This report aims at implementing, parallelizing, and evaluating the performance of a _5-point stencil heat equation solver_.
The exercise requires starting from a serial template and progressively introducing _MPI_ (for distributed memory parallelization) and _OpenMP_ (for shared memory parallelization).

Specifically, the objectives are:

- To design a correct parallel solver with proper boundary handling and energy source injection.
- To analyze performance and scalability of the implementation across different thread/task configurations.
- To compare _strong scaling_ and _weak scaling_ behaviors on the Leonardo supercomputer.

#line()

== Methodology

=== Heat Equation and Stencil Computation

- Governing equation: 2D heat diffusion with explicit finite-difference discretization.
- Five-point stencil approximation: new value at (i,j) depends on itself and its four immediate neighbors.
- Boundary conditions: periodic or fixed, depending on configuration.

=== Parallelization Strategy

- _MPI domain decomposition_: splitting the computational grid across MPI tasks.
- _OpenMP parallelization_: updating the local grid with multi-threading.
- _Communication model_: halo exchange between neighboring MPI tasks.
- _Instrumentation_: timers used to separately measure computation and communication overhead.

== Computational Environment

- _System_: Leonardo Supercomputer (DCGP partition).
- _Nodes_: Dual-socket CPUs, 56 cores/socket (112 cores/node).
- _Software stack_:

  - MPI implementation provided on Leonardo
  - OpenMP runtime from GCC/Intel compilers
  - SLURM workload manager (submission via `go_dcgp` batch script)

#line()

== Benchmarking & Scalability Studies

== OpenMP Scaling

- Single MPI task, thread count varied as {1,2,4,8,16,32,56,84,112}.
- Goal: assess shared-memory scalability, choose optimal threads-per-task ratio.

== MPI Scaling

- Based on OpenMP results, fix threads-per-task and scale number of MPI tasks.
- Strong scaling: problem size fixed, resources scaled {1,2,4,8,16,32 nodes}.
- Weak scaling: workload per core fixed, problem size scaled with number of nodes.

=== Metrics Collected

- Execution time per iteration.
- Communication/computation split.
- Speedup and efficiency:

  - *Speedup* $"Sp" = t(1) / t(n)$
  - *Efficiency* $"Eff" = "Sp" / n$

#line()

== Results & Analysis

- _OpenMP results_: plots showing speedup/efficiency across thread counts.
- _Strong scaling results_: speedup vs ideal scaling, efficiency drop with more nodes.
- _Weak scaling results_: runtime stability with increasing nodes, efficiency losses due to communication.
- _Discussion_: identify bottlenecks (communication overhead, load imbalance, boundary conditions).

#line()

== Conclusion

- The stencil solver scales well up to X nodes, but efficiency declines beyond Y due to communication costs.
- OpenMP parallelization achieves good performance up to Z threads before diminishing returns.
- MPI scaling shows that problem size must be sufficiently large to achieve efficiency.
- The implementation highlights trade-offs between communication overhead and computation parallelism.

#line()

== Project Scope

This report focuses on:

- Implementing a hybrid MPI+OpenMP stencil solver.
- Documenting job submission and instrumentation strategies.
- Presenting strong/weak scaling results on Leonardo.
- Analyzing performance trends in relation to theoretical models (Amdahl’s and Gustafson’s laws).
