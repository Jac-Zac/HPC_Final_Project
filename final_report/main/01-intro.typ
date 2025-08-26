#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Introduction
== Objective

This report aims at evaluating and comparing the performance of virtual machines (VMs) and containers in a controlled environment. Specifically, we use VirtualBox #footnote()[https://www.virtualbox.org] @virtualbox to deploy VMs and Docker #footnote()[https://www.docker.com] @docker to run containers, both operating Ubuntu Server #footnote()[https://ubuntu.com/download/server] @ubuntu-server.

The experiment is structured into two phases: first, setting up a cluster of VMs and a parallel cluster of containers with equivalent resource constraints; second, executing a series of benchmarks to measure and analyze their performance.

== Benchmark Overview

The following tools and methods were used to assess different performance dimensions: 

- *CPU & Memory*:
  - High-Performance Linpack (HPL) and the HPC Challenge (HPCC) suite
  - `sysbench` and `stress-ng` for general system load testing
- *Disk I/O*:
  - `IOZone` and for local and shared disk performance
- *Network*:
  - `iperf3` for measuring throughput and latency between nodes

== Host configuration 

The host computer has the following specifications: 
- *OS*: macOS Sequoia 15.4.1.

- *CPU*: Apple M4 Pro 12c/12t (8 Performance + 4 Efficiency). Base/Boost clock speed: ~3.4/4.5 GHz (P-cores). Each core thread corresponds to one core.

- *RAM*: 24 GB Unified LPDDR5X. Fully shared between CPU/GPU Neural Engine.

- *Disk*: Apple NVMe SSD 512GB SSD.

- VBoxManage --version 7.1.8r168469
-  Docker version 28.1.1, build 4eba377327

== Project Scope

*This report focuses on:*

- Documenting the setup process for both virtualised and containerized clusters
- Presenting benchmark results for each test case
- Analyzing and comparing the outcomes across environments

#todobox()[
The full set of benchmark data and scripts used in this analysis is available on #link("https://github.com/Jac-Zac/Cloud-Computing-Final-Exam")[GitHub], with custom README.md for each essentially each part of the process.
]
