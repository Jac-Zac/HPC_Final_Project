#!/usr/bin/env python3
"""
HPC Stencil Scaling Analysis Plotter

Generates scalability plots for MPI and OpenMP implementations of a stencil code.
This script creates strong scaling (Speedup, Efficiency), weak scaling (Efficiency),
and time breakdown (Computation vs. Communication) plots.

Usage:
    python plot_scaling.py --mpi-strong <file> --omp-strong <file> --mpi-weak <file> [options]

Examples:
    # Plot all three data files and save to 'scaling_plots/' directory
    python plot_scaling.py --mpi-strong mpi_strong.csv --omp-strong omp_strong.csv --mpi-weak mpi_weak.csv

    # Plot only MPI strong scaling and show the plot without saving
    python plot_scaling.py --mpi-strong mpi_strong.csv --no-save

    # Save plots without displaying them on screen (for remote servers)
    python plot_scaling.py --mpi-strong mpi_strong.csv --no-show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Apply a professional and clear style for the plots
plt.style.use("ggplot")


def load_data(file_path, resource_col):
    """Loads and validates the CSV data from the given path."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
    try:
        df = pd.read_csv(file_path)
        # Ensure required columns are present
        required_cols = [resource_col, "TotalTime"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: File '{file_path}' is missing one of the required columns: {required_cols}"
            )
            return None
        # Sort by the number of resources (threads/tasks)
        df = df.sort_values(by=resource_col).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error reading or parsing file '{file_path}': {e}")
        return None


def plot_strong_scaling(df, resource_col, label, save_dir, show):
    """Generates Speedup and Efficiency plots for strong scaling."""
    # --- Calculate Speedup and Efficiency ---
    baseline_time = df["TotalTime"].iloc[0]
    baseline_resources = df[resource_col].iloc[0]

    df["Speedup"] = baseline_time / df["TotalTime"]
    df["IdealSpeedup"] = df[resource_col] / baseline_resources
    df["Efficiency"] = (df["Speedup"] / df["IdealSpeedup"]) * 100

    # --- Plot Speedup ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        df[resource_col],
        df["Speedup"],
        marker="o",
        linestyle="-",
        label=f"Measured {label}",
    )
    ax.plot(
        df[resource_col],
        df["IdealSpeedup"],
        marker="",
        linestyle="--",
        color="black",
        label="Ideal Speedup",
    )

    ax.set_title(f"{label} Strong Scaling Speedup")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{label.lower()}_strong_speedup.png"))
    if show:
        plt.show()
    plt.close(fig)

    # --- Plot Efficiency ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df[resource_col], df["Efficiency"], marker="o", linestyle="-")
    ax.axhline(100, linestyle="--", color="black", label="Ideal Efficiency (100%)")

    ax.set_title(f"{label} Strong Scaling Efficiency")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(0, 110)  # Set Y-axis from 0% to 110%
    ax.legend()
    ax.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{label.lower()}_strong_efficiency.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_weak_scaling(df, resource_col, label, save_dir, show):
    """Generates an Efficiency plot for weak scaling."""
    baseline_time = df["TotalTime"].iloc[0]
    df["Efficiency"] = (baseline_time / df["TotalTime"]) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df[resource_col], df["Efficiency"], marker="o", linestyle="-")
    ax.axhline(100, linestyle="--", color="black", label="Ideal Efficiency (100%)")

    ax.set_title(f"{label} Weak Scaling Efficiency")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{label.lower()}_weak_efficiency.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_time_breakdown(df, resource_col, label, save_dir, show):
    """Creates a stacked bar chart of computation vs. communication time."""
    if "MaxCompTime" not in df.columns or "MaxCommTime" not in df.columns:
        print(
            f"Skipping time breakdown for {label}: missing 'MaxCompTime' or 'MaxCommTime' columns."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.6

    # Create the stacked bars
    ax.bar(df[resource_col], df["MaxCompTime"], width, label="Computation")
    ax.bar(
        df[resource_col],
        df["MaxCommTime"],
        width,
        bottom=df["MaxCompTime"],
        label="Communication",
    )

    ax.set_title(f"{label} Time Breakdown (Computation vs. Communication)")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(df[resource_col])
    ax.set_xticklabels(df[resource_col])
    ax.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{label.lower()}_time_breakdown.png"))
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="HPC Stencil Scaling Analysis Plotter",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Plot all three scaling types from specified files
  python %(prog)s --mpi-strong mpi_s.csv --omp-strong omp_s.csv --mpi-weak mpi_w.csv

  # Plot only MPI strong scaling data and do not save the plots
  python %(prog)s --mpi-strong mpi_strong_results.csv --no-save
""",
    )
    parser.add_argument(
        "--mpi-strong", type=str, help="Path to MPI strong scaling data file."
    )
    parser.add_argument(
        "--omp-strong", type=str, help="Path to OpenMP strong scaling data file."
    )
    parser.add_argument(
        "--mpi-weak", type=str, help="Path to MPI weak scaling data file."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="scaling_plots",
        help="Directory to save plots (default: 'scaling_plots').",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots interactively."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save plots to disk."
    )

    args = parser.parse_args()

    # Determine save directory and whether to show plots
    save_dir = None if args.no_save else args.save_dir
    show_plots = not args.no_show

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plots will be saved to '{save_dir}/'")

    # --- Process and Plot MPI Strong Scaling ---
    if args.mpi_strong:
        print(f"\nProcessing MPI Strong Scaling: {args.mpi_strong}")
        df_mpi_s = load_data(args.mpi_strong, resource_col="Tasks")
        if df_mpi_s is not None:
            plot_strong_scaling(df_mpi_s, "Tasks", "MPI", save_dir, show_plots)
            plot_time_breakdown(df_mpi_s, "Tasks", "MPI Strong", save_dir, show_plots)

    # --- Process and Plot OpenMP Strong Scaling ---
    if args.omp_strong:
        print(f"\nProcessing OpenMP Strong Scaling: {args.omp_strong}")
        df_omp_s = load_data(args.omp_strong, resource_col="Threads")
        if df_omp_s is not None:
            plot_strong_scaling(df_omp_s, "Threads", "OpenMP", save_dir, show_plots)

    # --- Process and Plot MPI Weak Scaling ---
    if args.mpi_weak:
        print(f"\nProcessing MPI Weak Scaling: {args.mpi_weak}")
        df_mpi_w = load_data(args.mpi_weak, resource_col="Tasks")
        if df_mpi_w is not None:
            plot_weak_scaling(df_mpi_w, "Tasks", "MPI", save_dir, show_plots)

    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
