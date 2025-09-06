#!/usr/bin/env python3
"""
Generate visualizations and animations from simulation data.
Used by the Makefile 'visualize' target.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
from stencil_utils import assemble_global_grid_from_patches, load_sources_from_logs

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.animation as animation
import matplotlib.pyplot as plt

print("Generating visualizations...")

# Parameters
ntasks = 4
grid_size = 100

# Load sources with correct coordinate transformation
sources = load_sources_from_logs("data_logging", ntasks=ntasks, grid_size=grid_size)
print(f"Loaded {len(sources)} sources: {sources}")

# Skip static images - focus on animation only

# Generate animation
print("Generating animation...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load all iterations
    frames = []
    max_iterations = 250  # Animation length

    for i in range(max_iterations):
        try:
            grid = assemble_global_grid_from_patches(
                prefix="data_logging/",
                iteration=i,
                ntasks=ntasks,
                grid_size=grid_size,
            )
            frames.append(grid[1:-1, 1:-1].copy())  # Remove halo
        except:
            break

    if len(frames) > 0:
        # Create animation
        im = ax.imshow(
            frames[0],
            cmap="hot",
            origin="lower",
            vmin=0,
            vmax=max(f.max() for f in frames),
        )
        ax.set_title("Heat Stencil Diffusion Animation")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Energy")

        # Plot sources
        for x, y in sources:
            ax.plot(
                x - 1,
                y - 1,
                "bo",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=1,
                label="Energy Source" if sources.index((x, y)) == 0 else "",
            )

        def animate(frame_num):
            im.set_array(frames[frame_num])
            ax.set_title(f"Heat Stencil Diffusion - Iteration {frame_num}")
            return [im]

        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=200, blit=True
        )
        anim.save("heat_diffusion_animation.gif", writer="pillow", fps=15)
        print(f"✓ Generated animation with {len(frames)} frames")
    else:
        print("✗ No frames found for animation")

except Exception as e:
    print(f"✗ Error generating animation: {e}")

print("Visualization generation complete!")
