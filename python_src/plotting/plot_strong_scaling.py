#!/usr/bin/env python3
"""
HPC Heat-Stencil Scaling Analysis Plotter

This script analyzes and plots scaling performance for MPI and OpenMP implementations
of the heat-stencil simulation. It generates multiple plots comparing measured performance
against ideal scaling curves, including a combined log-log plot with computation,
communication, and total time.

Usage:
    python plot_strong_scaling.py [options] [mpi_file] [openmp_file]

Options:
    --save-dir DIR              Save plots to specified directory (default: plots/)
    --no-show                   Don't display plots (useful for headless environments)
    --mpi-only                  Plot only MPI scaling data
    --openmp-only               Plot only OpenMP scaling data
    --combined-timing-only      Generate only the combined timing log-log plot

Examples:
    python plot_strong_scaling.py                          # Use default files
    python plot_strong_scaling.py --save-dir results/     # Save to custom directory
    python plot_strong_scaling.py mpi_data.csv openmp_data.csv
    python plot_strong_scaling.py --mpi-only mpi_data.csv
    python plot_strong_scaling.py --combined-timing-only mpi_data.csv openmp_data.csv
"""

import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install matplotlib numpy pandas")
    sys.exit(1)


def _ideal_time_curve(tasks, baseline_tasks, baseline_time):
    """Ideal strong-scaling time: T(n) = T(baseline) * (baseline_tasks / n)."""
    return [baseline_time * (baseline_tasks / float(n)) for n in tasks]


def _plot_comparison_with_ideal(
    mpi_data=None,
    openmp_data=None,
    ylabel="",
    title="",
    logy=True,
    save_dir=None,
    show_plot=True,
):
    """Compare scaling curves with ideal lines. Use log-log for time plots."""

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = False

    if mpi_data is not None and not mpi_data["values"].dropna().empty:
        mpi_tasks = mpi_data["Tasks"].to_list()
        mpi_values = mpi_data["values"].to_list()
        if all(v > 0 for v in mpi_values if pd.notna(v)):
            mpi_baseline_tasks = mpi_tasks[0]
            mpi_baseline_time = mpi_values[0]
            mpi_ideal = _ideal_time_curve(
                mpi_tasks, mpi_baseline_tasks, mpi_baseline_time
            )
            ax.plot(
                mpi_tasks, mpi_values, marker="o", label="MPI Measured", color="blue"
            )
            ax.plot(
                mpi_tasks,
                mpi_ideal,
                linestyle="--",
                alpha=0.7,
                label="MPI Ideal (1/n)",
                color="blue",
            )
            plotted = True

    if openmp_data is not None and not openmp_data["values"].dropna().empty:
        openmp_tasks = openmp_data["Threads"].to_list()
        openmp_values = openmp_data["values"].to_list()
        if all(v > 0 for v in openmp_values if pd.notna(v)):
            openmp_baseline_tasks = openmp_tasks[0]
            openmp_baseline_time = openmp_values[0]
            openmp_ideal = _ideal_time_curve(
                openmp_tasks, openmp_baseline_tasks, openmp_baseline_time
            )
            ax.plot(
                openmp_tasks,
                openmp_values,
                marker="s",
                label="OpenMP Measured",
                color="red",
            )
            ax.plot(
                openmp_tasks,
                openmp_ideal,
                linestyle="--",
                alpha=0.7,
                label="OpenMP Ideal (1/n)",
                color="red",
            )
            plotted = True

    if not plotted:
        return

    ax.set_xscale("log", base=2)

    # Only use log y-axis for time plots (where it makes sense)
    if logy and plotted:
        ax.set_yscale("log")

    ax.set_xlabel("Number of Tasks/Threads (log₂)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title.lower().replace(' ', '_').replace(':', '')}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_speedup_comparison(
    mpi_data=None, openmp_data=None, save_dir=None, show_plot=True
):
    """Compare speedup for MPI vs OpenMP with ideal speedup."""

    if mpi_data is None and openmp_data is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if mpi_data is not None:
        mpi_baseline_time = float(mpi_data.iloc[0]["TotalTime"])
        mpi_speedup = mpi_baseline_time / mpi_data["TotalTime"]
        mpi_tasks = mpi_data["Tasks"]
        mpi_ideal_speedup = mpi_tasks / mpi_tasks.iloc[0]

        ax.plot(
            mpi_tasks,
            mpi_speedup,
            marker="o",
            linestyle="-",
            label="MPI Speedup",
            color="blue",
        )
        ax.plot(
            mpi_tasks,
            mpi_ideal_speedup,
            linestyle="--",
            alpha=0.7,
            label="MPI Ideal",
            color="blue",
        )

    if openmp_data is not None:
        openmp_baseline_time = float(openmp_data.iloc[0]["TotalTime"])
        openmp_speedup = openmp_baseline_time / openmp_data["TotalTime"]
        openmp_threads = openmp_data["Threads"]
        openmp_ideal_speedup = openmp_threads / openmp_threads.iloc[0]

        ax.plot(
            openmp_threads,
            openmp_speedup,
            marker="s",
            linestyle="-",
            label="OpenMP Speedup",
            color="red",
        )
        ax.plot(
            openmp_threads,
            openmp_ideal_speedup,
            linestyle="--",
            alpha=0.7,
            label="OpenMP Ideal",
            color="red",
        )

    # Log-log makes sense for speedup - both axes have wide ranges
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Tasks/Threads (log₂)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Strong Scaling: Speedup Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "strong_scaling_speedup_comparison.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_combined_timing_loglog(
    mpi_data=None, openmp_data=None, save_dir=None, show_plot=True
):
    """Create a single log-log plot with computation, communication, and total time."""

    if mpi_data is None and openmp_data is None:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    plotted = False

    # Plot MPI data
    if mpi_data is not None:
        mpi_tasks = mpi_data["Tasks"]

        # Total time
        if "TotalTime" in mpi_data.columns and not mpi_data["TotalTime"].dropna().empty:
            ax.plot(
                mpi_tasks,
                mpi_data["TotalTime"],
                marker="o",
                linestyle="-",
                label="MPI Total Time",
                color="blue",
                linewidth=2,
            )
            plotted = True

        # Computation time
        if "MaxCompTime" in mpi_data.columns and not mpi_data["MaxCompTime"].dropna().empty:
            ax.plot(
                mpi_tasks,
                mpi_data["MaxCompTime"],
                marker="s",
                linestyle="--",
                label="MPI Computation Time",
                color="blue",
                alpha=0.8,
            )

        # Communication time
        if "MaxCommTime" in mpi_data.columns and not mpi_data["MaxCommTime"].dropna().empty:
            ax.plot(
                mpi_tasks,
                mpi_data["MaxCommTime"],
                marker="^",
                linestyle=":",
                label="MPI Communication Time",
                color="blue",
                alpha=0.8,
            )

    # Plot OpenMP data
    if openmp_data is not None:
        openmp_threads = openmp_data["Threads"]

        # Total time
        if "TotalTime" in openmp_data.columns and not openmp_data["TotalTime"].dropna().empty:
            ax.plot(
                openmp_threads,
                openmp_data["TotalTime"],
                marker="o",
                linestyle="-",
                label="OpenMP Total Time",
                color="red",
                linewidth=2,
            )
            plotted = True

        # Computation time
        if "MaxCompTime" in openmp_data.columns and not openmp_data["MaxCompTime"].dropna().empty:
            ax.plot(
                openmp_threads,
                openmp_data["MaxCompTime"],
                marker="s",
                linestyle="--",
                label="OpenMP Computation Time",
                color="red",
                alpha=0.8,
            )

        # Communication time
        if "MaxCommTime" in openmp_data.columns and not openmp_data["MaxCommTime"].dropna().empty:
            ax.plot(
                openmp_threads,
                openmp_data["MaxCommTime"],
                marker="^",
                linestyle=":",
                label="OpenMP Communication Time",
                color="red",
                alpha=0.8,
            )

    if not plotted:
        return

    # Log-log scaling
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Tasks/Threads (log₂)")
    ax.set_ylabel("Time (s) - log scale")
    ax.set_title("Strong Scaling: Combined Timing Analysis (Log-Log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "combined_timing_loglog.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_efficiency_comparison(
    mpi_data=None, openmp_data=None, save_dir=None, show_plot=True
):
    """Compare efficiency for MPI vs OpenMP. Linear y-axis for percentages."""

    if mpi_data is None and openmp_data is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if mpi_data is not None:
        mpi_baseline_time = float(mpi_data.iloc[0]["TotalTime"])
        mpi_speedup = mpi_baseline_time / mpi_data["TotalTime"]
        mpi_tasks = mpi_data["Tasks"]
        mpi_ideal_speedup = mpi_tasks / mpi_tasks.iloc[0]
        mpi_efficiency = 100.0 * mpi_speedup / mpi_ideal_speedup

        ax.plot(
            mpi_tasks,
            mpi_efficiency,
            marker="o",
            linestyle="-",
            label="MPI Efficiency",
            color="blue",
        )

    if openmp_data is not None:
        openmp_baseline_time = float(openmp_data.iloc[0]["TotalTime"])
        openmp_speedup = openmp_baseline_time / openmp_data["TotalTime"]
        openmp_threads = openmp_data["Threads"]
        openmp_ideal_speedup = openmp_threads / openmp_threads.iloc[0]
        openmp_efficiency = 100.0 * openmp_speedup / openmp_ideal_speedup

        ax.plot(
            openmp_threads,
            openmp_efficiency,
            marker="s",
            linestyle="-",
            label="OpenMP Efficiency",
            color="red",
        )

    ax.axhline(
        100.0, linestyle="--", alpha=0.7, label="Ideal Efficiency (100%)", color="black"
    )

    # Efficiency: linear y-axis makes most sense (0-100%)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Tasks/Threads (log₂)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Strong Scaling: Efficiency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot if directory specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "strong_scaling_efficiency_comparison.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def load_and_validate_data(file_path, expected_columns):
    """Load CSV data and validate required columns exist (flexible)."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")

    # Mandatory: TotalTime + either Tasks or Threads
    mandatory = ["TotalTime"]
    if "Tasks" in expected_columns:
        mandatory.append("Tasks")
    elif "Threads" in expected_columns:
        mandatory.append("Threads")

    missing_mandatory = [col for col in mandatory if col not in data.columns]
    if missing_mandatory:
        raise ValueError(
            f"Missing mandatory columns in {file_path}: {missing_mandatory}"
        )

    # Fill optional columns with NaN
    for col in expected_columns:
        if col not in data.columns:
            data[col] = np.nan

    # Convert numerics
    for col in expected_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=mandatory)
    sort_col = "Tasks" if "Tasks" in data.columns else "Threads"
    data = data.sort_values(sort_col)
    return data


def plot_scaling_comparison(
    mpi_file=None, openmp_file=None, save_dir=None, show_plot=True
):
    """Read MPI and/or OpenMP results and create comparison plots."""
    mpi_data = None
    openmp_data = None

    mpi_columns = ["Tasks", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"]
    openmp_columns = [
        "Threads",
        "TotalTime",
        "MaxCompTime",
        "MaxCommTime",
        "EnergyCompTime",
    ]

    if mpi_file:
        print(f"Loading MPI data from {mpi_file}...")
        mpi_data = load_and_validate_data(mpi_file, mpi_columns)

    if openmp_file:
        print(f"Loading OpenMP data from {openmp_file}...")
        openmp_data = load_and_validate_data(openmp_file, openmp_columns)

    if mpi_data is None and openmp_data is None:
        raise ValueError("No data provided: need at least MPI or OpenMP file.")

    # Plot different timing metrics
    timing_metrics = [
        ("TotalTime", "Total Time (s)", "Total Time Scaling", True),
        ("MaxCompTime", "Max Computation Time (s)", "Computation Time Scaling", True),
        (
            "MaxCommTime",
            "Max Communication Time (s)",
            "Communication Time Scaling",
            False,
        ),
        ("EnergyCompTime", "Energy Computation Time (s)", "Energy Time Scaling", True),
    ]

    for column, ylabel, title, logy in timing_metrics:
        if (mpi_data is not None and column in mpi_data.columns) or (
            openmp_data is not None and column in openmp_data.columns
        ):

            mpi_plot_data = None
            if mpi_data is not None and column in mpi_data.columns:
                mpi_plot_data = mpi_data.copy()
                mpi_plot_data["values"] = mpi_plot_data[column]

            openmp_plot_data = None
            if openmp_data is not None and column in openmp_data.columns:
                openmp_plot_data = openmp_data.copy()
                openmp_plot_data["values"] = openmp_plot_data[column]

            _plot_comparison_with_ideal(
                mpi_plot_data,
                openmp_plot_data,
                ylabel=ylabel,
                title=title,
                logy=logy,
                save_dir=save_dir,
                show_plot=show_plot,
            )

    # 5) Speedup - smart log scaling
    _plot_speedup_comparison(
        mpi_data, openmp_data, save_dir=save_dir, show_plot=show_plot
    )

    # 6) Combined timing log-log plot
    _plot_combined_timing_loglog(
        mpi_data, openmp_data, save_dir=save_dir, show_plot=show_plot
    )

    # 7) Efficiency - always linear y-axis
    _plot_efficiency_comparison(
        mpi_data, openmp_data, save_dir=save_dir, show_plot=show_plot
    )

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    if mpi_data is not None:
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

    if openmp_data is not None:
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
    import argparse

    parser = argparse.ArgumentParser(
        description="HPC Heat-Stencil Scaling Analysis Plotter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_strong_scaling.py                          # Use default files
  python plot_strong_scaling.py --save-dir results/     # Save to custom directory
  python plot_strong_scaling.py mpi_data.csv openmp_data.csv
  python plot_strong_scaling.py --mpi-only mpi_data.csv
  python plot_strong_scaling.py --combined-timing-only mpi_data.csv openmp_data.csv
  python plot_strong_scaling.py --no-show --save-dir plots/
        """,
    )

    parser.add_argument("files", nargs="*", help="MPI and/or OpenMP data files")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: plots/)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (useful for headless environments)",
    )
    parser.add_argument(
        "--mpi-only", action="store_true", help="Plot only MPI scaling data"
    )
    parser.add_argument(
        "--openmp-only", action="store_true", help="Plot only OpenMP scaling data"
    )
    parser.add_argument(
        "--combined-timing-only",
        action="store_true",
        help="Generate only the combined timing log-log plot with computation, communication, and total time"
    )

    args = parser.parse_args()

    # Determine which files to use
    mpi_file, openmp_file = None, None

    if args.mpi_only and args.openmp_only:
        parser.error("--mpi-only and --openmp-only cannot be used together")

    if args.combined_timing_only and (args.mpi_only or args.openmp_only):
        parser.error("--combined-timing-only cannot be used with --mpi-only or --openmp-only")

    if args.files:
        for file in args.files:
            if "mpi" in file.lower() or args.mpi_only:
                mpi_file = file
            elif "openmp" in file.lower() or args.openmp_only:
                openmp_file = file
            else:
                # If not specified, try to guess based on filename
                if "mpi" in file.lower():
                    mpi_file = file
                else:
                    openmp_file = file
    else:
        # Default files
        if not args.mpi_only:
            mpi_file = "mpi_results.txt"
        if not args.openmp_only:
            openmp_file = "openmp_scaling.txt"

    # Set default save directory if saving is requested
    save_dir = args.save_dir
    if save_dir is None and not args.no_show:
        save_dir = "plots"

    if args.combined_timing_only:
        # Load data and plot only combined timing
        mpi_data = None
        openmp_data = None

        mpi_columns = ["Tasks", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"]
        openmp_columns = [
            "Threads",
            "TotalTime",
            "MaxCompTime",
            "MaxCommTime",
            "EnergyCompTime",
        ]

        if mpi_file:
            print(f"Loading MPI data from {mpi_file}...")
            mpi_data = load_and_validate_data(mpi_file, mpi_columns)

        if openmp_file:
            print(f"Loading OpenMP data from {openmp_file}...")
            openmp_data = load_and_validate_data(openmp_file, openmp_columns)

        if mpi_data is None and openmp_data is None:
            raise ValueError("No data provided: need at least MPI or OpenMP file.")

        _plot_combined_timing_loglog(
            mpi_data, openmp_data, save_dir=save_dir, show_plot=not args.no_show
        )
    else:
        plot_scaling_comparison(
            mpi_file, openmp_file, save_dir=save_dir, show_plot=not args.no_show
        )


if __name__ == "__main__":
    main()
