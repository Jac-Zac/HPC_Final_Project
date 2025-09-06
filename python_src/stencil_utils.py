"""
Shared utilities for HPC Heat-Stencil simulation
Contains functions for loading sources, assembling global grids, and data processing
"""

import glob
import os
import re
import numpy as np


def load_sources_from_logs(log_dir="data_logging", ntasks=4, grid_size=100):
    """Load all sources from data_logging/sources_rank*.txt files and convert to global coordinates."""
    sources = []

    # Determine decomposition based on ntasks
    if ntasks == 1:
        ranks_per_row = 1
        ranks_per_col = 1
    elif ntasks == 2:
        ranks_per_row = 2
        ranks_per_col = 1
    elif ntasks == 4:
        ranks_per_row = 2
        ranks_per_col = 2
    else:
        import math
        ranks_per_row = int(math.sqrt(ntasks))
        ranks_per_col = ntasks // ranks_per_row

    # Calculate patch sizes (same logic as C code)
    patch_size_x = grid_size // ranks_per_row
    patch_size_y = grid_size // ranks_per_col

    for fname in sorted(glob.glob(os.path.join(log_dir, "sources_rank*.txt"))):
        # Extract rank number from filename
        rank_match = re.search(r'sources_rank(\d+)\.txt', fname)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))

        # Calculate rank position in the grid
        rank_x = rank % ranks_per_row
        rank_y = rank // ranks_per_row

        # Calculate global offset for this rank
        start_x = rank_x * patch_size_x
        start_y = rank_y * patch_size_y

        # Handle extra cells for uneven division (same as C code)
        extra_x = rank_x if rank_x < (grid_size % ranks_per_row) else (grid_size % ranks_per_row)
        extra_y = rank_y if rank_y < (grid_size % ranks_per_col) else (grid_size % ranks_per_col)

        start_x += extra_x
        start_y += extra_y

        with open(fname) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    # Convert local coordinates to global coordinates
                    local_x, local_y = map(int, parts)
                    global_x = start_x + local_x
                    global_y = start_y + local_y
                    sources.append((global_x, global_y))

    return sources


def read_global_bin(filename, nx, ny):
    """Read the global binary produced by your MPI C code."""
    data = np.fromfile(filename, dtype=np.float64)
    return data.reshape((ny, nx))


def assemble_global_grid_from_patches(
    prefix="data_parallel/", iteration=0, ntasks=4, grid_size=100
):
    """Assemble global grid from individual rank patches for testing."""
    global_grid = np.zeros((grid_size + 2, grid_size + 2))

    # Determine decomposition based on ntasks
    if ntasks == 1:
        ranks_per_row = 1
        ranks_per_col = 1
    elif ntasks == 2:
        ranks_per_row = 2
        ranks_per_col = 1
    elif ntasks == 4:
        ranks_per_row = 2
        ranks_per_col = 2
    else:
        # For other numbers, try to find a reasonable decomposition
        import math
        ranks_per_row = int(math.sqrt(ntasks))
        ranks_per_col = ntasks // ranks_per_row

    # Calculate base patch sizes
    base_patch_size_x = grid_size // ranks_per_row
    base_patch_size_y = grid_size // ranks_per_col

    for rank in range(ntasks):
        filename = f"{prefix}{rank}_plane_{iteration:05d}.bin"
        try:
            # Read the local patch data (interior only, no halos)
            patch_data = np.fromfile(filename, dtype=np.float64)

            # Calculate actual patch size for this rank (accounting for remainder)
            rank_x = rank % ranks_per_row
            rank_y = rank // ranks_per_row

            # Apply same logic as C code: first r ranks get extra cell
            patch_size_x = base_patch_size_x + (1 if rank_x < (grid_size % ranks_per_row) else 0)
            patch_size_y = base_patch_size_y + (1 if rank_y < (grid_size % ranks_per_col) else 0)

            patch = patch_data.reshape((patch_size_y, patch_size_x))

            # Calculate starting position in global grid
            start_x = 1  # +1 for halo
            start_y = 1  # +1 for halo

            # Add up all previous ranks' sizes
            for prev_rank in range(rank):
                prev_rank_x = prev_rank % ranks_per_row
                prev_rank_y = prev_rank // ranks_per_row
                prev_size_x = base_patch_size_x + (1 if prev_rank_x < (grid_size % ranks_per_row) else 0)
                prev_size_y = base_patch_size_y + (1 if prev_rank_y < (grid_size % ranks_per_col) else 0)

                if prev_rank_x == rank_x:
                    start_y += prev_size_y
                if prev_rank_y == rank_y:
                    start_x += prev_size_x

            # Copy interior data to global grid (no halos in patch data)
            for j in range(patch_size_y):
                for i in range(patch_size_x):
                    global_y = start_y + j
                    global_x = start_x + i
                    if global_y < grid_size + 2 and global_x < grid_size + 2:
                        global_grid[global_y, global_x] = patch[j, i]

        except FileNotFoundError:
            print(f"Warning: Could not find {filename}")
            continue
        except Exception as e:
            print(f"Warning: Error processing {filename}: {e}")
            continue

    return global_grid


def extract_energies_from_bins(prefix="plane_global_", nx=100, ny=100):
    """Read all iteration files and return total energies like C would print."""
    if prefix == "plane_":
        # For serial files, exclude global files
        files = sorted(
            [f for f in glob.glob(f"{prefix}*.bin") if "global" not in f],
            key=lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0,
        )
    else:
        # For global files, use normal pattern
        files = sorted(
            glob.glob(f"{prefix}*.bin"),
            key=lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0,
        )
    energies = []
    for f in files:
        grid = read_global_bin(f, nx, ny)
        # sum interior only, matching C get_total_energy
        energy = np.sum(grid[1:-1, 1:-1])
        energies.append(energy)
    return energies


def visualize_grid(grid, sources=None, title="Heat Stencil Grid", save_path=None):
    """
    Visualize the heat stencil grid with optional source locations.

    Args:
        grid: 2D numpy array of the grid data
        sources: List of (x, y) tuples for source locations
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the grid
        im = ax.imshow(grid[1:-1, 1:-1], cmap='hot', origin='lower')
        ax.set_title(title)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Energy')

        # Plot sources if provided
        if sources:
            for x, y in sources:
                # Convert to plot coordinates (matplotlib uses row, col)
                circle = Circle((x-1, y-1), 0.5, color='blue', alpha=0.7, label='Energy Source' if sources.index((x,y)) == 0 else "")
                ax.add_patch(circle)

            if sources:
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")
        print(f"Grid shape: {grid.shape}")
        print(f"Max energy: {np.max(grid[1:-1, 1:-1]):.6f}")
        print(f"Total energy: {np.sum(grid[1:-1, 1:-1]):.6f}")
        if sources:
            print(f"Sources: {sources}")


def load_and_visualize_iteration(iteration=0, ntasks=4, grid_size=100, sources=None,
                                prefix="data_logging/", save_path=None):
    """
    Load and visualize a specific iteration from MPI patches.

    Args:
        iteration: Iteration number to visualize
        ntasks: Number of MPI tasks
        grid_size: Size of the global grid
        sources: Source locations (if None, will try to load from logs)
        prefix: Prefix for data files
        save_path: Path to save visualization
    """
    if sources is None:
        sources = load_sources_from_logs("data_logging", ntasks=ntasks, grid_size=grid_size)

    grid = assemble_global_grid_from_patches(
        prefix=prefix, iteration=iteration, ntasks=ntasks, grid_size=grid_size
    )

    title = f"Heat Stencil - Iteration {iteration}"
    visualize_grid(grid, sources, title, save_path)

    return grid