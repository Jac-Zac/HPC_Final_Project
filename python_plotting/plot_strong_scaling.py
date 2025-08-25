#!/usr/bin/env python

import math
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _ideal_time_curve(tasks, baseline_tasks, baseline_time):
    """Ideal strong-scaling time: T(n) = T(baseline) * (baseline_tasks / n)."""
    return [baseline_time * (baseline_tasks / float(n)) for n in tasks]


def _plot_time_with_ideal(x, y, ylabel, title, logy=True):
    """Generic time vs tasks plot with ideal line."""
    # Sort by tasks (just in case)
    df = pd.DataFrame({"Tasks": x, "Measured": y}).sort_values("Tasks")
    tasks = df["Tasks"].to_list()
    measured = df["Measured"].to_list()

    # Baseline = first (smallest) task count
    baseline_tasks = tasks[0]
    baseline_time = measured[0]
    ideal = _ideal_time_curve(tasks, baseline_tasks, baseline_time)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tasks, measured, marker="o", linestyle="-", label="Measured")
    ax.plot(tasks, ideal, linestyle="--", label="Ideal (1/n)")

    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Tasks (log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_scaling(file_path="results_scaling.txt"):
    """Read scaling results CSV and plot strong-scaling views with ideal lines."""
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {e}")

    # Coerce numerics
    for col in ["Tasks", "TotalTime", "MaxCompTime", "MaxCommTime", "EnergyCompTime"]:
        if col not in data.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna().sort_values("Tasks")

    tasks = data["Tasks"].to_list()

    # ---------- 1) Total time (log–log) with ideal 1/n ----------
    _plot_time_with_ideal(
        tasks,
        data["TotalTime"].to_list(),
        ylabel="Total Time (s)",
        title="Strong Scaling: Total Time vs Tasks (with Ideal 1/n)",
        logy=True,
    )

    # ---------- 2) Max computation time (log–log) with ideal 1/n ----------
    _plot_time_with_ideal(
        tasks,
        data["MaxCompTime"].to_list(),
        ylabel="Max Computation Time (s)",
        title="Strong Scaling: Max Computation Time vs Tasks (with Ideal 1/n)",
        logy=True,
    )

    # ---------- 3) Max communication time (log–log) with ideal 1/n ----------
    # Note: communication rarely follows 1/n perfectly, but we still overlay it as a reference.
    _plot_time_with_ideal(
        tasks,
        data["MaxCommTime"].to_list(),
        ylabel="Max Communication Time (s)",
        title="Strong Scaling: Max Communication Time vs Tasks (with Ideal 1/n)",
        logy=True,
    )

    # ---------- 4) Energy computation time (log–log) with ideal 1/n ----------
    _plot_time_with_ideal(
        tasks,
        data["EnergyCompTime"].to_list(),
        ylabel="Energy Computation Time (s)",
        title="Strong Scaling: Energy Computation Time vs Tasks (with Ideal 1/n)",
        logy=True,
    )

    # ---------- 5) Speedup with ideal = n ----------
    # Baseline: smallest Tasks
    baseline_row = data.iloc[0]
    baseline_tasks = int(baseline_row["Tasks"])
    t1 = float(baseline_row["TotalTime"])

    speedup = t1 / data["TotalTime"]
    ideal_speedup = data["Tasks"] / baseline_tasks

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["Tasks"], speedup, marker="s", linestyle="-", label="Measured Speedup")
    ax.plot(data["Tasks"], ideal_speedup, linestyle="--", label="Ideal Speedup (= n)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Tasks (log2)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Strong Scaling: Speedup vs Tasks (with Ideal n)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---------- 6) Efficiency with ideal = 1 ----------
    efficiency = speedup / ideal_speedup  # = speedup / (n / baseline)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        data["Tasks"],
        100.0 * efficiency,
        marker="^",
        linestyle="-",
        label="Measured Efficiency",
    )
    ax.axhline(100.0, linestyle="--", label="Ideal Efficiency (= 100%)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Tasks (log2)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Strong Scaling: Efficiency vs Tasks (with Ideal 100%)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "results_scaling.txt"
    plot_scaling(file_path)


if __name__ == "__main__":
    main()
