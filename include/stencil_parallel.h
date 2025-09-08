#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

#define STENCIL_CENTER_COEFF 0.5
#define STENCIL_NEIGHBOR_COEFF 0.125 // 1/8

// 64 bytes = 512 bits alignment for SIMD
#define MEMORY_ALIGNMENT 64

typedef unsigned int uint;

// Direction enum for halo exchanges
typedef enum { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 } direction_t;

// Error codes enum
typedef enum {
  SUCCESS = 0,
  ERROR_INVALID_GRID_SIZE = 1,
  ERROR_INVALID_NUM_SOURCES = 2,
  ERROR_INVALID_NUM_ITERATIONS = 3,
  ERROR_NULL_POINTER = 4,
  ERROR_MEMORY_ALLOCATION = 5,
  ERROR_INITIALIZE_SOURCES = 6,
  ERROR_MPI_FAILURE = 7,
  ERROR_FILE_DUMPING = 8,
  ERROR_INVALID_ENERGY_VALUE = 9
} error_code_t;

typedef uint vec2_t[2];

typedef struct {
  // restrict makes sure that the pointers are the only references
  // to the memory they point to
  double *restrict data;
  vec2_t size;
} plane_t;

extern void inject_energy(const int, const int, const vec2_t *, const double,
                          plane_t *, const vec2_t);

extern void get_total_energy(plane_t *, double *);

extern error_code_t initialize(MPI_Comm *, int, int, int, char **, vec2_t *,
                               vec2_t *, int *, int *, int *, int *, int *,
                               int *, vec2_t **, double *, plane_t *);

extern error_code_t exchange_halos(plane_t *plane, int *neighbours,
                                   MPI_Comm *Comm, MPI_Request requests[8],
                                   MPI_Datatype north_south_type,
                                   MPI_Datatype east_west_type);

error_code_t memory_release(plane_t *, MPI_Datatype *, MPI_Datatype *);

error_code_t output_energy_stat(int, plane_t *, double, int, MPI_Comm *);

extern error_code_t dump(const double *data, const uint size[2],
                         const char *filename);

inline void inject_energy(const int periodic, const int num_sources,
                          const vec2_t *sources, const double energy,
                          plane_t *plane, const vec2_t mpi_tasks_grid) {
  register const uint size_x = plane->size[_x_] + 2;
  double *restrict data = plane->data;

#define IDX(i, j) ((j) * size_x + (i))
  for (int s = 0; s < num_sources; s++) {
    // Local x and y coordinate
    uint x = sources[s][_x_];
    uint y = sources[s][_y_];

    // Add energy to the source location
    data[IDX(x, y)] += energy;

    if (periodic) {
      // HACK: I need to review this

      if (mpi_tasks_grid[_x_] == 1) {
        if (x == 1) {
          data[IDX(plane->size[_x_] + 1, y)] += energy;
        }
        if (x == plane->size[_x_]) {
          data[IDX(0, y)] += energy;
        }
      }

      if (mpi_tasks_grid[_y_] == 1) {
        if (y == 1) {
          data[IDX(x, plane->size[_y_] + 1)] += energy;
        }

        if (y == plane->size[_y_]) {
          data[IDX(x, 0)] += energy;
        }
      }
    }
  }
#undef IDX
}

extern error_code_t exchange_halos(plane_t *plane, int *neighbours,
                                   MPI_Comm *Comm, MPI_Request requests[8],
                                   MPI_Datatype north_south_type,
                                   MPI_Datatype east_west_type) {
  int rc = MPI_SUCCESS; // Accumulate MPI errors with OR to check all at once

  const uint size_x = plane->size[_x_];
  const uint size_y = plane->size[_y_];
  const uint stride = size_x + 2;

  // Sets SEND buffers to point to internal plane data and RECV buffers to halo
  // locations for direct MPI datatype-based communication (avoids copies)

  // Send addresses (point to internal data that will be sent to neighbors)
  // First element of top internal row (skip halo)
  double *send_north = &plane->data[stride + 1];
  // First element of bottom internal row
  double *send_south = &plane->data[size_y * stride + 1];
  // Leftmost internal column (skip left halo)
  double *send_west = &plane->data[stride + 1];
  // Rightmost internal column
  double *send_east = &plane->data[stride + size_x];

  // Receive addresses (point to halo locations where data will be stored)
  // Top halo row (ghost cells for north neighbor)
  double *recv_north = &plane->data[1];
  // Bottom halo row
  double *recv_south = &plane->data[(size_y + 1) * stride + 1];
  // Left halo column (skip top-left corner halo)
  double *recv_west = &plane->data[stride];
  // Right halo column
  double *recv_east = &plane->data[stride + size_x + 1];

  // NOTE: count = 1 because we send the entire pre-committed datatype
  // NOTE: MPI automatically handles sends/receives to MPI_PROC_NULL as no-ops
  // NOTE: Keep the receive first send after order we would use for blocking
  // version which avoid deadlocks
  rc |= MPI_Irecv(recv_south, 1, north_south_type, neighbours[SOUTH], NORTH,
                  *Comm, &requests[0]);
  rc |= MPI_Isend(send_north, 1, north_south_type, neighbours[NORTH], NORTH,
                  *Comm, &requests[1]);

  rc |= MPI_Irecv(recv_north, 1, north_south_type, neighbours[NORTH], SOUTH,
                  *Comm, &requests[2]);
  rc |= MPI_Isend(send_south, 1, north_south_type, neighbours[SOUTH], SOUTH,
                  *Comm, &requests[3]);

  // EAST-WEST exchanges
  rc |= MPI_Irecv(recv_west, 1, east_west_type, neighbours[WEST], EAST, *Comm,
                  &requests[4]);
  rc |= MPI_Isend(send_east, 1, east_west_type, neighbours[EAST], EAST, *Comm,
                  &requests[5]);

  rc |= MPI_Irecv(recv_east, 1, east_west_type, neighbours[EAST], WEST, *Comm,
                  &requests[6]);
  rc |= MPI_Isend(send_west, 1, east_west_type, neighbours[WEST], WEST, *Comm,
                  &requests[7]);

  // Single check at the end
  if (rc != MPI_SUCCESS) {
    return ERROR_MPI_FAILURE;
  }

  return SUCCESS;
}

void inline update_plane_inner(const plane_t *old_plane, plane_t *new_plane) {
  const double *restrict old = old_plane->data;
  double *restrict newp = new_plane->data;

  const uint stride =
      old_plane->size[_x_] + 2; // leading dimension (with halos)
  const uint x_size = old_plane->size[_x_];
  const uint y_size = old_plane->size[_y_];

  // Tunable tile sizes — experiment with 128, 256, 512
  const uint Tx = 256;
  const uint Ty = 256;

#pragma omp parallel for collapse(2) schedule(static)
  for (uint jj = 2; jj <= y_size - 1; jj += Ty) {
    for (uint ii = 2; ii <= x_size - 1; ii += Tx) {

      uint jmax = (jj + Ty - 1 < y_size - 1) ? jj + Ty - 1 : y_size - 1;
      uint imax = (ii + Tx - 1 < x_size - 1) ? ii + Tx - 1 : x_size - 1;

      for (uint j = jj; j <= jmax; ++j) {
        const uint row = j * stride;

#pragma omp simd
        for (uint i = ii; i <= imax; ++i) {
          newp[row + i] =
              0.25 * (old[row + i - 1] + old[row + i + 1] +
                      old[(j - 1) * stride + i] + old[(j + 1) * stride + i]);
        }
      }
    }
  }
}

// inline void update_plane_inner(const plane_t *old_plane, plane_t *new_plane)
// {
//   const uint f_xsize = old_plane->size[_x_] + 2;
//   const uint x_size = old_plane->size[_x_];
//   const uint y_size = old_plane->size[_y_];
//
//   double *restrict new_p = new_plane->data;
//   const double *restrict old_p = old_plane->data;
//
//   // 5-point stencil coefficients: center + 4 neighbors (N,S,E,W)
//   const double c_center = STENCIL_CENTER_COEFF;
//   const double c_neigh = STENCIL_NEIGHBOR_COEFF;
//
//   // Outer loop: parallelize over rows for load balancing
//   // Pre-compute row pointers outside inner loop reduce address calculations
// #pragma omp parallel for schedule(static)
//   for (uint j = 2; j <= y_size - 1; ++j) {
//     // Pre-compute row pointers to avoid repeated offset calculations
//     const double *row_above = old_p + (j - 1) * f_xsize;
//     const double *row_center = old_p + j * f_xsize;
//     const double *row_below = old_p + (j + 1) * f_xsize;
//     double *row_new = new_p + j * f_xsize;
//
//     // Inner loop: automatically vectorized by -O3 due to simple arithmetic
//     // No conditional branches to maintain SIMD efficiency
// #pragma omp simd
//     for (uint i = 2; i <= x_size - 1; ++i) {
//
//       // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
//       //       if this patch is at some border without periodic conditions;
//       //       in that case it is assumed that the +-1 points are outside the
//       //       plate and always have a value of 0, i.e. they are an
//       //       "infinite sink" of heat
//       const double center = row_center[i];
//       const double left = row_center[i - 1];
//       const double right = row_center[i + 1];
//       const double up = row_above[i];
//       const double down = row_below[i];
//
//       // 5-point stencil computation: center*0.5 + neighbors*0.125 each
//       row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
//     }
//   }
// }

#define UPDATED
#ifdef UPDATED

inline void update_plane_borders(const int periodic, const vec2_t N,
                                 const plane_t *old_plane, plane_t *new_plane) {

#define IDX(i, j) ((j) * f_xsize + (i))
  const uint f_xsize = old_plane->size[_x_] + 2;
  const uint x_size = old_plane->size[_x_];
  const uint y_size = old_plane->size[_y_];

  double *restrict new_p = new_plane->data;
  const double *restrict old_p = old_plane->data;

  const double c_center = STENCIL_CENTER_COEFF;
  const double c_neigh = STENCIL_NEIGHBOR_COEFF;

  /* --- Update left and right border columns (excluding corners) --- */
#pragma omp parallel for schedule(static)
  for (uint j = 2; j < y_size; ++j) {
    // left border (i=1)
    {
      const uint i = 1;
      new_p[IDX(i, j)] = old_p[IDX(i, j)] * c_center +
                         (old_p[IDX(i - 1, j)] + old_p[IDX(i + 1, j)] +
                          old_p[IDX(i, j - 1)] + old_p[IDX(i, j + 1)]) *
                             c_neigh;
    }
    // right border (i=x_size)
    {
      const uint i = x_size;
      new_p[IDX(i, j)] = old_p[IDX(i, j)] * c_center +
                         (old_p[IDX(i - 1, j)] + old_p[IDX(i + 1, j)] +
                          old_p[IDX(i, j - 1)] + old_p[IDX(i, j + 1)]) *
                             c_neigh;
    }
  }

  /* --- Update top and bottom border rows (including corners) --- */
#pragma omp parallel for schedule(static)
  for (uint i = 1; i <= x_size; ++i) {
    // top row (j=1)
    new_p[IDX(i, 1)] = old_p[IDX(i, 1)] * c_center +
                       (old_p[IDX(i - 1, 1)] + old_p[IDX(i + 1, 1)] +
                        old_p[IDX(i, 0)] + old_p[IDX(i, 2)]) *
                           c_neigh;

    // bottom row (j=y_size)
    new_p[IDX(i, y_size)] =
        old_p[IDX(i, y_size)] * c_center +
        (old_p[IDX(i - 1, y_size)] + old_p[IDX(i + 1, y_size)] +
         old_p[IDX(i, y_size - 1)] + old_p[IDX(i, y_size + 1)]) *
            c_neigh;
  }

  /* --- Periodic wrap for single-rank dimensions --- */
  if (periodic) {
    if (N[_x_] == 1) {
#pragma omp parallel for schedule(static)
      for (uint j = 1; j <= y_size; ++j) {
        new_p[IDX(0, j)] = new_p[IDX(x_size, j)];
        new_p[IDX(x_size + 1, j)] = new_p[IDX(1, j)];
      }
    }
    if (N[_y_] == 1) {
      double *row_topghost = &new_p[IDX(0, 0)];
      double *row_bottomghost = &new_p[IDX(0, y_size + 1)];
      double *row_top = &new_p[IDX(0, 1)];
      double *row_bottom = &new_p[IDX(0, y_size)];
#pragma omp parallel for schedule(static)
      for (uint i = 1; i <= x_size; ++i) {
        row_topghost[i] = row_bottom[i];
        row_bottomghost[i] = row_top[i];
      }
    }
  }

#undef IDX
}
#else
inline void update_plane_borders(const int periodic, const vec2_t N,
                                 const plane_t *old_plane, plane_t *new_plane) {
  const uint f_xsize = old_plane->size[_x_] + 2;
  const uint x_size = old_plane->size[_x_];
  const uint y_size = old_plane->size[_y_];

  double *restrict new_p = new_plane->data;
  const double *restrict old_p = old_plane->data;

  // 5-point stencil coefficients: center + 4 neighbors (N,S,E,W)
  const double c_center = STENCIL_CENTER_COEFF;
  const double c_neigh = STENCIL_NEIGHBOR_COEFF;

  /* --- Update left and right border columns (i=1 and i=x_size) --- */

#pragma omp parallel for schedule(static)
  for (uint j = 1; j <= y_size; ++j) {
    // Pre-compute row pointers to avoid repeated offset calculations
    const double *row_above = old_p + (j - 1) * f_xsize;
    const double *row_center = old_p + j * f_xsize;
    const double *row_below = old_p + (j + 1) * f_xsize;
    double *row_new = new_p + j * f_xsize;

    /* left border (i=1) */
    {
      const uint i = 1;
      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];
      row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
    }

    /* right border (i=x_size) */
    {
      const uint i = x_size;
      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];
      row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
    }
  }

  /* --- Update top and bottom border rows (j=1 and j=y_size) --- */
#pragma omp parallel for schedule(static)
  for (uint i = 1; i <= x_size; ++i) {
    /* top row (j=1) */
    {
      const uint j = 1;
      const double *row_above = old_p + (j - 1) * f_xsize;
      const double *row_center = old_p + j * f_xsize;
      const double *row_below = old_p + (j + 1) * f_xsize;
      double *row_new = new_p + j * f_xsize;

      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];
      row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
    }

    /* bottom row (j=y_size) */
    {
      const uint j = y_size;
      const double *row_above = old_p + (j - 1) * f_xsize;
      const double *row_center = old_p + j * f_xsize;
      const double *row_below = old_p + (j + 1) * f_xsize;
      double *row_new = new_p + j * f_xsize;

      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];
      row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
    }
  }

  // /* --- Periodic wrap for single-rank dimensions --- */
  // Periodic boundary handling for single-rank dimensions
  // When there's only one MPI rank in a dimension, we can't exchange halos
  // with neighboring processes, so we must wrap boundaries locally
  if (periodic) {
    // Wrap left/right edges to simulate periodic boundaries
    if (N[_x_] == 1) {
      for (uint j = 1; j <= y_size; ++j) {
        double *row = new_p + j * f_xsize;
        row[0] = row[x_size];
        row[x_size + 1] = row[1];
      }
    }
    // Wrap top/bottom edges to simulate periodic boundaries
    if (N[_y_] == 1) {
      double *row_top = new_p + 1 * f_xsize;
      double *row_bottom = new_p + y_size * f_xsize;
      double *row_topghost = new_p + 0 * f_xsize;
      double *row_bottomghost = new_p + (y_size + 1) * f_xsize;

      for (uint i = 1; i <= x_size; ++i) {
        row_topghost[i] = row_bottom[i];
        row_bottomghost[i] = row_top[i];
      }
    }
  }

  // // Periodic boundary handling for single-rank dimensions
  // // When there's only one MPI rank in a dimension, we can't exchange halos
  // // with neighboring processes, so we must wrap boundaries locally
  // if (periodic) {
  //   // Special case: single rank along X dimension
  //   // Wrap left/right edges to simulate periodic boundaries
  //   if (N[_x_] == 1) {
  //     for (uint j = 1; j <= y_size; ++j) {
  //       double *row = new_p + j * f_xsize;
  //       row[0] = row[x_size];
  //       row[x_size + 1] = row[1];
  //     }
  //   }
  //   // Special case: single rank along Y dimension
  //   // Wrap top/bottom edges to simulate periodic boundaries
  //   if (N[_y_] == 1) {
  //     double *row_top = new_p + 1 * f_xsize;
  //     double *row_bottom = new_p + y_size * f_xsize;
  //     double *row_topghost = new_p + 0 * f_xsize;
  //     double *row_bottomghost = new_p + (y_size + 1) * f_xsize;
  //
  //     for (uint i = 1; i <= x_size; ++i) {
  //       row_topghost[i] = row_bottom[i];
  //       row_bottomghost[i] = row_top[i];
  //     }
  //   }
  // }
}

#endif

inline void get_total_energy(plane_t *plane, double *energy) {
  register const int x_size = plane->size[_x_];
  register const int y_size = plane->size[_y_];
  register const int f_size = x_size + 2;

  double *restrict data = plane->data;

#if defined(LONG_ACCURACY)
  long double tot_energy = 0;
#else
  double tot_energy = 0;
#endif

#pragma omp parallel for reduction(+ : tot_energy)
  for (uint j = 1; j <= y_size; ++j) {
    const double *row = data + j * f_size;
    // Automatically hints the compiler to do simd reduction here
#pragma omp simd reduction(+ : tot_energy)
    for (uint i = 1; i <= x_size; ++i)
      tot_energy += row[i];
  }

  *energy = (double)tot_energy;
}
