#!/usr/bin/env python3
"""
HPC Stencil Scaling Analysis Plotter (Updated)

Generates scalability plots for MPI and OpenMP implementations of a stencil code.
This script creates strong scaling (Speedup, Efficiency), weak scaling (Efficiency),
and time breakdown (Computation vs. Communication).

Updates:
- X-axis uses log2 scale for resource counts (Threads/Tasks).
- Speedup plot is log-log by default.
- Each point is annotated with its value for better readability with smart placement.
- Skip communication breakdown plot for OpenMP.
- In time breakdown plots, communication values are annotated above bars and computation values just below their bar tops.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


# ---------------- Utility ----------------
def annotate_points(ax, x, y, flip_threshold=90):
    """Add text annotations near each plotted point with smart placement."""
    for xi, yi in zip(x, y):
        if yi > flip_threshold:
            offset = (0, -10)
            va = "top"
        else:
            offset = (0, 5)
            va = "bottom"
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            va=va,
            fontsize=8,
        )


def load_data(file_path, resource_col):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
    try:
        df = pd.read_csv(file_path)
        required_cols = [resource_col, "TotalTime"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: File '{file_path}' is missing one of the required columns: {required_cols}"
            )
            return None
        df = df.sort_values(by=resource_col).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error reading or parsing file '{file_path}': {e}")
        return None


# ---------------- Strong Scaling ----------------
def plot_strong_scaling(data_dict, resource_col, title_prefix, save_dir, show):
    if not data_dict:
        return

    # --- Speedup ---
    fig_speedup, ax_speedup = plt.subplots(figsize=(8, 6))
    ideal_plotted = False

    for label, df in data_dict.items():
        baseline_time = df["TotalTime"].iloc[0]
        baseline_resources = df[resource_col].iloc[0]

        df["Speedup"] = baseline_time / df["TotalTime"]
        ax_speedup.plot(
            df[resource_col],
            df["Speedup"],
            marker="o",
            linestyle="-",
            label=f"Measured {label}",
        )
        annotate_points(ax_speedup, df[resource_col], df["Speedup"])

        if not ideal_plotted:
            df["IdealSpeedup"] = df[resource_col] / baseline_resources
            ax_speedup.plot(
                df[resource_col],
                df["IdealSpeedup"],
                linestyle="--",
                color="black",
                label="Ideal Speedup",
            )
            ideal_plotted = True

    ax_speedup.set_title(f"{title_prefix} Strong Scaling Speedup")
    ax_speedup.set_xlabel(f"Number of {resource_col}")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_yscale("log")  # log-log scaling
    ax_speedup.set_xticks(data_dict[list(data_dict.keys())[0]][resource_col])
    ax_speedup.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_speedup.legend()
    ax_speedup.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{title_prefix.lower()}_strong_speedup.png")
        )
    if show:
        plt.show()
    plt.close(fig_speedup)

    # --- Efficiency ---
    fig_eff, ax_eff = plt.subplots(figsize=(8, 6))
    for label, df in data_dict.items():
        baseline_time = df["TotalTime"].iloc[0]
        baseline_resources = df[resource_col].iloc[0]
        speedup = baseline_time / df["TotalTime"]
        ideal_speedup = df[resource_col] / baseline_resources
        efficiency = (speedup / ideal_speedup) * 100
        ax_eff.plot(
            df[resource_col],
            efficiency,
            marker="o",
            linestyle="-",
            label=f"Efficiency {label}",
        )
        annotate_points(ax_eff, df[resource_col], efficiency)

    ax_eff.axhline(100, linestyle="--", color="black", label="Ideal (100%)")
    ax_eff.set_title(f"{title_prefix} Strong Scaling Efficiency")
    ax_eff.set_xlabel(f"Number of {resource_col}")
    ax_eff.set_ylabel("Efficiency (%)")
    ax_eff.set_ylim(0, 110)
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_xticks(data_dict[list(data_dict.keys())[0]][resource_col])
    ax_eff.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_eff.legend()
    ax_eff.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{title_prefix.lower()}_strong_efficiency.png")
        )
    if show:
        plt.show()
    plt.close(fig_eff)


# ---------------- Weak Scaling ----------------
def plot_weak_scaling(df, resource_col, label, save_dir, show):
    baseline_time = df["TotalTime"].iloc[0]
    df["Efficiency"] = (baseline_time / df["TotalTime"]) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df[resource_col], df["Efficiency"], marker="o", linestyle="-")
    annotate_points(ax, df[resource_col], df["Efficiency"])
    ax.axhline(100, linestyle="--", color="black", label="Ideal (100%)")

    ax.set_title(f"{label} Weak Scaling Efficiency")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Efficiency (%)")
    ax.set_ylim(0, 110)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df[resource_col])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{label.lower()}_weak_efficiency.png"))
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Time Breakdown ----------------
def plot_time_breakdown(data_dict, resource_col, title_prefix, save_dir, show):
    if not data_dict:
        return

    labels = list(data_dict.keys())
    n_datasets = len(labels)
    df_ref = data_dict[labels[0]]
    resources = df_ref[resource_col]
    x = np.arange(len(resources))

    total_width = 0.8
    bar_width = total_width / n_datasets

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (label, df) in enumerate(data_dict.items()):
        if "MaxCompTime" not in df.columns or "MaxCommTime" not in df.columns:
            print(f"Skipping time breakdown for {label}: missing required columns.")
            continue

        offset = (i - n_datasets / 2 + 0.5) * bar_width
        bars_comp = ax.bar(
            x + offset, df["MaxCompTime"], bar_width, label=f"Computation ({label})"
        )
        bars_comm = ax.bar(
            x + offset,
            df["MaxCommTime"],
            bar_width,
            bottom=df["MaxCompTime"],
            label=f"Communication ({label})",
        )

        # Annotate communication above total bar
        for b_comp, b_comm in zip(bars_comp, bars_comm):
            total_height = b_comp.get_height() + b_comm.get_height()
            comm_height = b_comm.get_height()
            if comm_height > 0:
                ax.annotate(
                    f"{comm_height:.2f}",
                    xy=(b_comm.get_x() + b_comm.get_width() / 2, total_height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="blue",
                )

        # Annotate computation just below its top
        for b in bars_comp:
            height = b.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(b.get_x() + b.get_width() / 2, height),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )

    ax.set_title(f"{title_prefix} Time Breakdown")
    ax.set_xlabel(f"Number of {resource_col}")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(resources)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{title_prefix.lower()}_time_breakdown.png")
        )
    if show:
        plt.show()
    plt.close(fig)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="HPC Stencil Scaling Analysis Plotter")
    parser.add_argument(
        "--mpi-strong", type=str, help="Path to MPI strong scaling data file."
    )
    parser.add_argument(
        "--omp-spread",
        type=str,
        help="Path to OpenMP strong scaling file (spread binding).",
    )
    parser.add_argument(
        "--omp-close",
        type=str,
        help="Path to OpenMP strong scaling file (close binding).",
    )
    parser.add_argument(
        "--mpi-weak", type=str, help="Path to MPI weak scaling data file."
    )
    parser.add_argument(
        "--save-dir", type=str, default="scaling_plots", help="Directory to save plots."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots interactively."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save plots to disk."
    )

    args = parser.parse_args()

    save_dir = None if args.no_save else args.save_dir
    show_plots = not args.no_show

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Plots will be saved to '{save_dir}/'")

    # MPI Strong
    if args.mpi_strong:
        print(f"\nProcessing MPI Strong Scaling: {args.mpi_strong}")
        df_mpi_s = load_data(args.mpi_strong, "Tasks")
        if df_mpi_s is not None:
            plot_strong_scaling({"MPI": df_mpi_s}, "Tasks", "MPI", save_dir, show_plots)
            plot_time_breakdown(
                {"MPI": df_mpi_s}, "Tasks", "MPI Strong", save_dir, show_plots
            )

    # OpenMP Strong (skip communication breakdown)
    omp_data_dict = {}
    if args.omp_spread:
        df_omp_spread = load_data(args.omp_spread, "Threads")
        if df_omp_spread is not None:
            omp_data_dict["Spread"] = df_omp_spread
    if args.omp_close:
        df_omp_close = load_data(args.omp_close, "Threads")
        if df_omp_close is not None:
            omp_data_dict["Close"] = df_omp_close

    if omp_data_dict:
        print("\nProcessing OpenMP Strong Scaling...")
        plot_strong_scaling(omp_data_dict, "Threads", "OpenMP", save_dir, show_plots)
        # no plot_time_breakdown for OpenMP

    # MPI Weak
    if args.mpi_weak:
        print(f"\nProcessing MPI Weak Scaling: {args.mpi_weak}")
        df_mpi_w = load_data(args.mpi_weak, "Tasks")
        if df_mpi_w is not None:
            plot_weak_scaling(df_mpi_w, "Tasks", "MPI", save_dir, show_plots)

    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
