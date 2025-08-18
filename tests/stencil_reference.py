import numpy as np


def inject_energy(periodic, sources, energy, grid):
    nx, ny = grid.shape
    for x, y in sources:
        grid[y, x] += energy
        if periodic:
            if x == 1:
                grid[y, nx - 1] += energy
            if x == nx - 2:
                grid[y, 0] += energy
            if y == 1:
                grid[ny - 1, x] += energy
            if y == ny - 2:
                grid[0, x] += energy
    return grid


def update_plane(periodic, old_grid):
    nx, ny = old_grid.shape
    new_grid = old_grid.copy()

    alpha = 0.6

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            c = old_grid[j, i]
            l, r = old_grid[j, i - 1], old_grid[j, i + 1]
            u, d = old_grid[j - 1, i], old_grid[j + 1, i]

            # EXACTLY same formula as in C
            result = alpha * c
            result += (1 - alpha) * (l + r) / 4.0
            result += (1 - alpha) * (u + d) / 4.0
            new_grid[j, i] = result

    if periodic:
        new_grid[0, :] = new_grid[-2, :]
        new_grid[-1, :] = new_grid[1, :]
        new_grid[:, 0] = new_grid[:, -2]
        new_grid[:, -1] = new_grid[:, 1]

    return new_grid


def total_energy(grid):
    return np.sum(grid)
