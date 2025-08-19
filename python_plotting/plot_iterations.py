#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd


def plot_timings(file_path="timings.log"):
    """Read a timings log file and plot computation and energy times separately."""
    # Read file
    try:
        data = pd.read_csv(file_path, sep=r"\s+", comment="#")  # fixed deprecation
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {e}")

    # Ensure numeric and drop invalid rows
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    # Skip the first iteration for plotting

    # NOTE: if you don't want to plot the first two iterations
    # data = data.iloc[2:]  # all rows except the first

    # Use a built-in style to avoid errors
    plt.style.use("ggplot")  # safe built-in style

    # Plot computation time
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        data["iter"],
        data["comp_time"],
        marker="o",
        markevery=max(1, len(data) // 20),
        linestyle="-",
        color="blue",
        label="Comp Time",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Computation Time per Iteration")
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # Plot energy time
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(
        data["iter"],
        data["energy_time"],
        marker="x",
        markevery=max(1, len(data) // 20),
        linestyle="--",
        color="red",
        label="Energy Time",
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Energy Time per Iteration")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def main():
    file_path = "timings.log"
    plot_timings(file_path)


if __name__ == "__main__":
    main()
