#!/usr/bin/env python3
"""
HPC Heat-Stencil Scaling Analysis Plotter

This script analyzes and plots scaling performance for MPI and OpenMP implementations
of the heat-stencil simulation. It generates multiple plots comparing measured performance
against ideal scaling curves.

Usage:
    python plot_strong_scaling.py [options] [mpi_file] [openmp_file]

Options:
    --save-dir DIR     Save plots to specified directory (default: plots/)
    --no-show         Don't display plots (useful for headless environments)
    --mpi-only        Plot only MPI scaling data
    --openmp-only     Plot only OpenMP scaling data

Examples:
    python plot_strong_scaling.py                          # Use default files
    python plot_strong_scaling.py --save-dir results/     # Save to custom directory
    python plot_strong_scaling.py mpi_data.csv openmp_data.csv
    python plot_strong_scaling.py --mpi-only mpi_data.csv
"""

import sys
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    if MATPLOTLIB_AVAILABLE:
        print("Warning: pandas/numpy not available. Install with: pip install pandas numpy")


def _ideal_time_curve(tasks, baseline_tasks, baseline_time):
    """Ideal strong-scaling time: T(n) = T(baseline) * (baseline_tasks / n)."""
    return [baseline_time * (baseline_tasks / float(n)) for n in tasks]


def _plot_comparison_with_ideal(
    mpi_data=None, openmp_data=None, ylabel="", title="", logy=True, save_dir=None, show_plot=True
):
    """Compare scaling curves with ideal lines. Use log-log for time plots."""
    if not MATPLOTLIB_AVAILABLE:
        print(f"Skipping plot '{title}' - matplotlib not available")
        return

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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_speedup_comparison(mpi_data=None, openmp_data=None, save_dir=None, show_plot=True):
    """Compare speedup for MPI vs OpenMP with ideal speedup."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping speedup plot - matplotlib not available")
        return

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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_efficiency_comparison(mpi_data=None, openmp_data=None, save_dir=None, show_plot=True):
    """Compare efficiency for MPI vs OpenMP. Linear y-axis for percentages."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping efficiency plot - matplotlib not available")
        return

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
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
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


def plot_scaling_comparison(mpi_file=None, openmp_file=None, save_dir=None, show_plot=True):
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

    print("\n=== SUMMARY STATISTICS ===")

    # 1) Total Time - usually benefits from log-log
    mpi_total = mpi_data.copy() if mpi_data is not None else None
    if mpi_total is not None:
        mpi_total["values"] = mpi_total["TotalTime"]
    openmp_total = openmp_data.copy() if openmp_data is not None else None
    if openmp_total is not None:
        openmp_total["values"] = openmp_total["TotalTime"]
    _plot_comparison_with_ideal(
        mpi_total, openmp_total, ylabel="Total Time (s)", title="Total Time Scaling",
        save_dir=save_dir, show_plot=show_plot
    )

    # 2) Max Computation Time
    if (mpi_data is not None and "MaxCompTime" in mpi_data.columns) or (
        openmp_data is not None and "MaxCompTime" in openmp_data.columns
    ):
        mpi_comp = mpi_data.copy() if mpi_data is not None else None
        if mpi_comp is not None:
            mpi_comp["values"] = mpi_comp["MaxCompTime"]
        openmp_comp = openmp_data.copy() if openmp_data is not None else None
        if openmp_comp is not None:
            openmp_comp["values"] = openmp_comp["MaxCompTime"]
        _plot_comparison_with_ideal(
            mpi_comp,
            openmp_comp,
            ylabel="Max Computation Time (s)",
            title="Computation Time Scaling",
            save_dir=save_dir, show_plot=show_plot
        )

    # 3) Max Communication Time - linear y-axis shows overhead growth
    if (mpi_data is not None and "MaxCommTime" in mpi_data.columns) or (
        openmp_data is not None and "MaxCommTime" in openmp_data.columns
    ):
        mpi_comm = mpi_data.copy() if mpi_data is not None else None
        if mpi_comm is not None:
            mpi_comm["values"] = mpi_comm["MaxCommTime"]
        openmp_comm = openmp_data.copy() if openmp_data is not None else None
        if openmp_comm is not None:
            openmp_comm["values"] = openmp_comm["MaxCommTime"]
        _plot_comparison_with_ideal(
            mpi_comm,
            openmp_comm,
            ylabel="Max Communication Time (s)",
            title="Communication Time Scaling",
            logy=False,  # Linear y-axis shows communication overhead better
            save_dir=save_dir, show_plot=show_plot
        )

    # 4) Energy Computation Time
    if (mpi_data is not None and "EnergyCompTime" in mpi_data.columns) or (
        openmp_data is not None and "EnergyCompTime" in openmp_data.columns
    ):
        mpi_energy = mpi_data.copy() if mpi_data is not None else None
        if mpi_energy is not None:
            mpi_energy["values"] = mpi_energy["EnergyCompTime"]
        openmp_energy = openmp_data.copy() if openmp_data is not None else None
        if openmp_energy is not None:
            openmp_energy["values"] = openmp_energy["EnergyCompTime"]
        _plot_comparison_with_ideal(
            mpi_energy,
            openmp_energy,
            ylabel="Energy Computation Time (s)",
            title="Energy Time Scaling",
            save_dir=save_dir, show_plot=show_plot
        )

    # 5) Speedup - smart log scaling
    _plot_speedup_comparison(mpi_data, openmp_data, save_dir=save_dir, show_plot=show_plot)

    # 6) Efficiency - always linear y-axis
    _plot_efficiency_comparison(mpi_data, openmp_data, save_dir=save_dir, show_plot=show_plot)

    # --- Summary ---
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
  python plot_strong_scaling.py --no-show --save-dir plots/
        """
    )

    parser.add_argument('files', nargs='*', help='MPI and/or OpenMP data files')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (default: plots/)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (useful for headless environments)')
    parser.add_argument('--mpi-only', action='store_true',
                       help='Plot only MPI scaling data')
    parser.add_argument('--openmp-only', action='store_true',
                       help='Plot only OpenMP scaling data')

    args = parser.parse_args()

    # Determine which files to use
    mpi_file, openmp_file = None, None

    if args.mpi_only and args.openmp_only:
        parser.error("--mpi-only and --openmp-only cannot be used together")

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

    plot_scaling_comparison(mpi_file, openmp_file, save_dir=save_dir, show_plot=not args.no_show)


def test_basic_functionality():
    """Test basic functionality without matplotlib."""
    print("Testing basic functionality...")

    if not PANDAS_AVAILABLE:
        print("✗ Pandas/numpy not available for data loading tests")
        return

    # Test data loading
    try:
        mpi_data = load_and_validate_data("mpi_results.txt", ["Tasks", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"])
        print(f"✓ MPI data loaded: {len(mpi_data)} rows")
    except Exception as e:
        print(f"✗ MPI data loading failed: {e}")

    try:
        openmp_data = load_and_validate_data("openmp_scaling.txt", ["Threads", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"])
        print(f"✓ OpenMP data loaded: {len(openmp_data)} rows")
    except Exception as e:
        print(f"✗ OpenMP data loading failed: {e}")

    print("Basic functionality test completed.")


if __name__ == "__main__":
    # If matplotlib is not available, provide helpful message and run basic tests
    if not MATPLOTLIB_AVAILABLE:
        print("\n" + "="*60)
        print("HPC Heat-Stencil Scaling Analysis Plotter")
        print("="*60)
        print("matplotlib not available. To install:")
        print("  pip install matplotlib pandas numpy")
        print("\nRunning basic functionality tests...")
        print("="*60 + "\n")

        test_basic_functionality()

        print("\n" + "="*60)
        print("Usage examples:")
        print("  python plot_strong_scaling.py")
        print("  python plot_strong_scaling.py --save-dir results/")
        print("  python plot_strong_scaling.py --no-show --save-dir plots/")
        print("="*60 + "\n")
        sys.exit(1)

    main()
