import numpy as np


def inject_energy(periodic, sources, energy, grid):
    # shape is (rows, cols) → (ny, nx)
    ny, nx = grid.shape
    xsize = nx - 2  # interior width  (matches C: size[_x_])
    ysize = ny - 2  # interior height (matches C: size[_y_])

    for x, y in sources:  # x is column index, y is row index (1..size)
        grid[y, x] += energy
        if periodic:
            # exact C behavior: duplicate into opposite ghost cells when on edges
            if x == 1:
                grid[y, xsize + 1] += energy  # right ghost col
            if x == xsize:
                grid[y, 0] += energy  # left  ghost col
            if y == 1:
                grid[ysize + 1, x] += energy  # bottom ghost row
            if y == ysize:
                grid[0, x] += energy  # top    ghost row
    return grid


def update_plane(periodic, old_grid):
    ny, nx = old_grid.shape  # (rows, cols)
    new_grid = old_grid.copy()

    alpha = 0.6
    beta = (1.0 - alpha) * 0.25

    # interior update (1..ysize, 1..xsize)
    for j in range(1, ny - 1):  # rows
        for i in range(1, nx - 1):  # cols
            c = old_grid[j, i]
            l, r = old_grid[j, i - 1], old_grid[j, i + 1]
            u, d = old_grid[j - 1, i], old_grid[j + 1, i]
            new_grid[j, i] = alpha * c + beta * (l + r + u + d)

    if periodic:
        xsize = nx - 2
        ysize = ny - 2
        # match C: copy only interior spans into the ghost cells
        for i in range(1, xsize + 1):
            new_grid[0, i] = new_grid[ysize, i]  # top ghost   ← last interior row
            new_grid[ysize + 1, i] = new_grid[1, i]  # bottom ghost← first interior row
        for j in range(1, ysize + 1):
            new_grid[j, 0] = new_grid[j, xsize]  # left  ghost ← last interior col
            new_grid[j, xsize + 1] = new_grid[j, 1]  # right ghost ← first interior col

    return new_grid


def total_energy(grid):
    # sum interior only, like C get_total_energy (i=1..xsize, j=1..ysize)
    return np.sum(grid[1:-1, 1:-1])
