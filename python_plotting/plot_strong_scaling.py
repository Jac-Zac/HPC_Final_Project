#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ideal_time_curve(tasks, baseline_tasks, baseline_time):
    """Ideal strong-scaling time: T(n) = T(baseline) * (baseline_tasks / n)."""
    return [baseline_time * (baseline_tasks / float(n)) for n in tasks]


def _plot_comparison_with_ideal(mpi_data, openmp_data, ylabel, title, logy=True):
    """Compare MPI vs OpenMP with ideal scaling lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # MPI data
    mpi_tasks = mpi_data["Tasks"].to_list()
    mpi_values = mpi_data["values"].to_list()
    mpi_baseline_tasks = mpi_tasks[0]
    mpi_baseline_time = mpi_values[0]
    mpi_ideal = _ideal_time_curve(mpi_tasks, mpi_baseline_tasks, mpi_baseline_time)

    # OpenMP data (using Threads column)
    openmp_tasks = openmp_data["Threads"].to_list()
    openmp_values = openmp_data["values"].to_list()
    openmp_baseline_tasks = openmp_tasks[0]
    openmp_baseline_time = openmp_values[0]
    openmp_ideal = _ideal_time_curve(
        openmp_tasks, openmp_baseline_tasks, openmp_baseline_time
    )

    # Plot measured values
    ax.plot(
        mpi_tasks,
        mpi_values,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="MPI Measured",
        color="blue",
    )
    ax.plot(
        openmp_tasks,
        openmp_values,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="OpenMP Measured",
        color="red",
    )

    # Plot ideal lines
    ax.plot(
        mpi_tasks,
        mpi_ideal,
        linestyle="--",
        alpha=0.7,
        label="MPI Ideal (1/n)",
        color="blue",
    )
    ax.plot(
        openmp_tasks,
        openmp_ideal,
        linestyle="--",
        alpha=0.7,
        label="OpenMP Ideal (1/n)",
        color="red",
    )

    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Tasks/Threads (log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_speedup_comparison(mpi_data, openmp_data):
    """Compare speedup for MPI vs OpenMP with ideal speedup."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # MPI speedup
    mpi_baseline_time = float(mpi_data.iloc[0]["TotalTime"])
    mpi_speedup = mpi_baseline_time / mpi_data["TotalTime"]
    mpi_tasks = mpi_data["Tasks"]
    mpi_ideal_speedup = mpi_tasks / mpi_tasks.iloc[0]

    # OpenMP speedup
    openmp_baseline_time = float(openmp_data.iloc[0]["TotalTime"])
    openmp_speedup = openmp_baseline_time / openmp_data["TotalTime"]
    openmp_threads = openmp_data["Threads"]
    openmp_ideal_speedup = openmp_threads / openmp_threads.iloc[0]

    # Plot measured speedup
    ax.plot(
        mpi_tasks,
        mpi_speedup,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="MPI Speedup",
        color="blue",
    )
    ax.plot(
        openmp_threads,
        openmp_speedup,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="OpenMP Speedup",
        color="red",
    )

    # Plot ideal speedup
    ax.plot(
        mpi_tasks,
        mpi_ideal_speedup,
        linestyle="--",
        alpha=0.7,
        label="Ideal Speedup",
        color="black",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Tasks/Threads (log2)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Strong Scaling: Speedup Comparison (MPI vs OpenMP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_efficiency_comparison(mpi_data, openmp_data):
    """Compare efficiency for MPI vs OpenMP."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # MPI efficiency
    mpi_baseline_time = float(mpi_data.iloc[0]["TotalTime"])
    mpi_speedup = mpi_baseline_time / mpi_data["TotalTime"]
    mpi_tasks = mpi_data["Tasks"]
    mpi_ideal_speedup = mpi_tasks / mpi_tasks.iloc[0]
    mpi_efficiency = 100.0 * mpi_speedup / mpi_ideal_speedup

    # OpenMP efficiency
    openmp_baseline_time = float(openmp_data.iloc[0]["TotalTime"])
    openmp_speedup = openmp_baseline_time / openmp_data["TotalTime"]
    openmp_threads = openmp_data["Threads"]
    openmp_ideal_speedup = openmp_threads / openmp_threads.iloc[0]
    openmp_efficiency = 100.0 * openmp_speedup / openmp_ideal_speedup

    # Plot efficiency
    ax.plot(
        mpi_tasks,
        mpi_efficiency,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="MPI Efficiency",
        color="blue",
    )
    ax.plot(
        openmp_threads,
        openmp_efficiency,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="OpenMP Efficiency",
        color="red",
    )

    # Ideal efficiency line
    ax.axhline(
        100.0, linestyle="--", alpha=0.7, label="Ideal Efficiency (100%)", color="black"
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Tasks/Threads (log2)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Strong Scaling: Efficiency Comparison (MPI vs OpenMP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_and_validate_data(file_path, expected_columns):
    """Load CSV data and validate required columns exist."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")

    # Check required columns
    missing_cols = [col for col in expected_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    # Coerce numerics and drop NaN rows
    for col in expected_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()

    # Sort by Tasks/Threads column
    sort_col = "Tasks" if "Tasks" in data.columns else "Threads"
    data = data.sort_values(sort_col)

    return data


def plot_scaling_comparison(
    mpi_file="mpi_results.txt", openmp_file="openmp_scaling.txt"
):
    """Read both MPI and OpenMP results and create comparison plots."""

    # Define expected columns for each file type
    mpi_columns = ["Tasks", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"]
    openmp_columns = [
        "Threads",
        "TotalTime",
        "MaxCompTime",
        "MaxCommTime",
        "EnergyCompTime",
    ]

    # Load and validate data
    print(f"Loading MPI data from {mpi_file}...")
    mpi_data = load_and_validate_data(mpi_file, mpi_columns)

    print(f"Loading OpenMP data from {openmp_file}...")
    openmp_data = load_and_validate_data(openmp_file, openmp_columns)

    print(f"MPI data points: {len(mpi_data)}")
    print(f"OpenMP data points: {len(openmp_data)}")

    # Create comparison plots

    # 1) Total Time Comparison
    mpi_total = mpi_data.copy()
    mpi_total["values"] = mpi_total["TotalTime"]
    openmp_total = openmp_data.copy()
    openmp_total["values"] = openmp_total["TotalTime"]

    _plot_comparison_with_ideal(
        mpi_total,
        openmp_total,
        ylabel="Total Time (s)",
        title="Strong Scaling Comparison: Total Time (MPI vs OpenMP)",
        logy=True,
    )

    # 2) Max Computation Time Comparison
    mpi_comp = mpi_data.copy()
    mpi_comp["values"] = mpi_comp["MaxCompTime"]
    openmp_comp = openmp_data.copy()
    openmp_comp["values"] = openmp_comp["MaxCompTime"]

    _plot_comparison_with_ideal(
        mpi_comp,
        openmp_comp,
        ylabel="Max Computation Time (s)",
        title="Strong Scaling Comparison: Max Computation Time (MPI vs OpenMP)",
        logy=True,
    )

    # 3) Max Communication Time Comparison
    mpi_comm = mpi_data.copy()
    mpi_comm["values"] = mpi_comm["MaxCommTime"]
    openmp_comm = openmp_data.copy()
    openmp_comm["values"] = openmp_comm["MaxCommTime"]

    _plot_comparison_with_ideal(
        mpi_comm,
        openmp_comm,
        ylabel="Max Communication Time (s)",
        title="Strong Scaling Comparison: Max Communication Time (MPI vs OpenMP)",
        logy=True,
    )

    # 4) Energy Computation Time Comparison
    mpi_energy = mpi_data.copy()
    mpi_energy["values"] = mpi_energy["EnergyCompTime"]
    openmp_energy = openmp_data.copy()
    openmp_energy["values"] = openmp_energy["EnergyCompTime"]

    _plot_comparison_with_ideal(
        mpi_energy,
        openmp_energy,
        ylabel="Energy Computation Time (s)",
        title="Strong Scaling Comparison: Energy Computation Time (MPI vs OpenMP)",
        logy=True,
    )

    # 5) Speedup Comparison
    _plot_speedup_comparison(mpi_data, openmp_data)

    # 6) Efficiency Comparison
    _plot_efficiency_comparison(mpi_data, openmp_data)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")

    # MPI stats
    mpi_max_speedup = (mpi_data.iloc[0]["TotalTime"] / mpi_data["TotalTime"]).max()
    mpi_max_efficiency = (
        100.0
        * mpi_max_speedup
        / (mpi_data["Tasks"].iloc[-1] / mpi_data["Tasks"].iloc[0])
    )

    print(
        f"MPI - Max speedup: {mpi_max_speedup:.2f}x at {mpi_data['Tasks'].iloc[-1]} tasks"
    )
    print(f"MPI - Best efficiency: {mpi_max_efficiency:.1f}% at max scaling")

    # OpenMP stats
    openmp_max_speedup = (
        openmp_data.iloc[0]["TotalTime"] / openmp_data["TotalTime"]
    ).max()
    openmp_best_idx = (
        openmp_data.iloc[0]["TotalTime"] / openmp_data["TotalTime"]
    ).idxmax()
    openmp_best_threads = openmp_data.loc[openmp_best_idx, "Threads"]
    openmp_best_efficiency = (
        100.0
        * openmp_max_speedup
        / (openmp_best_threads / openmp_data["Threads"].iloc[0])
    )

    print(
        f"OpenMP - Max speedup: {openmp_max_speedup:.2f}x at {openmp_best_threads} threads"
    )
    print(
        f"OpenMP - Best efficiency: {openmp_best_efficiency:.1f}% at {openmp_best_threads} threads"
    )


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) == 3:
        mpi_file, openmp_file = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        mpi_file, openmp_file = "mpi_results.txt", "openmp_scaling.txt"
    else:
        print("Usage: python script.py [mpi_file openmp_file]")
        print(
            "Default: python script.py  (uses mpi_results.txt and openmp_scaling.txt)"
        )
        sys.exit(1)

    plot_scaling_comparison(mpi_file, openmp_file)


if __name__ == "__main__":
    main()
