/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <float.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CPU_TIME                                                               \
  ({                                                                           \
    struct timespec ts;                                                        \
    clock_gettime(CLOCK_REALTIME, &ts),                                        \
        (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;                         \
  })

#define CPU_TIME_th                                                            \
  ({                                                                           \
    struct timespec myts;                                                      \
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &myts),                             \
        (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9;                     \
  })

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

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
  ERROR_MPI_FAILURE = 7
} error_code_t;

// Stencil coefficients
#define ALPHA 0.5

// ============================================================
//
// function prototypes

int initialize(int, char **, uint *, int *, int *, int *, int **, double *,
               double **, int *);

int memory_release(double *, int *);

extern int inject_energy(const int, const int, const int *, const double,
                         const uint[2], double *);

extern void update_plane(const int, const uint[2], const double *, double *);

extern void get_total_energy(const uint[2], const double *, double *);

// ============================================================
//
// function definition for inline functions

inline int inject_energy(const int periodic, const int num_sources,
                         const int *sources, const double energy,
                         const uint my_size[2], double *plane) {
#define IDX(i, j) ((j) * (my_size[_x_] + 2) + (i))
  for (int s = 0; s < num_sources; s++) {

    uint x = sources[2 * s];
    uint y = sources[2 * s + 1];
    plane[IDX(x, y)] += energy;

    if (periodic) {
      if (x == 1)
        plane[IDX(my_size[_x_] + 1, y)] += energy;
      if (x == my_size[_x_])
        plane[IDX(0, y)] += energy;
      if (y == 1)
        plane[IDX(x, my_size[_y_] + 1)] += energy;
      if (y == my_size[_y_])
        plane[IDX(x, 0)] += energy;
    }
  }
#undef IDX

  return 0;
}

/*
 * calculate the new energy values
 * the old plane contains the current data, the new plane
 * will store the updated data
 *
 * NOTE: in parallel, every MPI task will perform the
 *       calculation for its patch
 */
inline void update_plane(const int periodic, const uint size[2],
                         const double *old_points, double *new_points) {

  // clang: ISO C++17 does not allow 'register' storage class specifier
  const int f_xsize = size[_x_] + 2;
  const int x_size = size[_x_];
  const int y_size = size[_y_];

  // Use defined stencil coefficients
  const double alpha = ALPHA;
  const double beta = (1.0 - ALPHA) * 0.25; // (1-alpha)/4

// NOTE: loop unrolling doesn't seem to increase performance
// #pragma omp parallel for collapse(2)
// #pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(static)
  for (int j = 1; j <= y_size; j++) {
    const double *row_above = old_points + (j - 1) * f_xsize;
    const double *row_center = old_points + j * f_xsize;
    const double *row_below = old_points + (j + 1) * f_xsize;
    double *row_new = new_points + j * f_xsize;

    // NOTE: Parallelizing this loop instead makes it super

    // NOTE: This inner loop is automatically vectorized by the compiler I
    // checked the assembly The slow part is memory read and write
    // You can force to not vectorized it by -fno-tree-vectorize and need it
    // doesn't vectorize but the speed difference is very minor
    // TODO: Improve memory read and write hear
    for (int i = 1; i <= x_size; i++) {
      // Five-point stencil
      // five-points stencil formula

      // however the implicit methods are not stable
      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];

      row_new[i] = center * alpha + beta * (left + right + up + down);
    }

    // simpler stencil with no explicit diffusivity
    // always conserve the smoothed quantity
    // alpha here mimics how much "easily" the heat travels
    // NOTE: Professor left this comment I'm not really sure what I have to do
    // with it if it should be implemented or anything
    //
    // implantation from the derivation of
    // 3-points 2nd order derivatives
    // however, that should depends on an adaptive
    // time-stepping so that given a diffusivity
    // coefficient the amount of energy diffused is
    // "small"
    /*
      double alpha = alpha_guess;
      double sum = old_points[IDX(i, j)];

      int done = 0;
      do {
        double sum_i = alpha * (old_points[IDX(i - 1, j)] +
                                old_points[IDX(i + 1, j)] - 2 * sum);
        double sum_j = alpha * (old_points[IDX(i, j - 1)] +
                                old_points[IDX(i, j + 1)] - 2 * sum);
        result = sum + (sum_i + sum_j);
        double ratio = fabs((result - sum) / (sum != 0 ? sum : 1.0));
        done = ((ratio < 2.0) && (result >= 0)); // not too fast diffusion and
                                                 // not so fast that the(i, j)
goes below zero energy alpha /= 2; } while (!done);
      */
  }

  /*
   * propagate boundaries if they are periodic
   *
   * NOTE: when is that needed in distributed memory, if any?
   */
  // Periodic boundary propagation
  if (periodic) {
    // Top & bottom wrap
    double *row_top = new_points + 1 * f_xsize;
    double *row_bottom = new_points + y_size * f_xsize;
    double *row_topghost = new_points + 0 * f_xsize;
    double *row_bottomghost = new_points + (y_size + 1) * f_xsize;

    for (int i = 1; i <= x_size; i++) {
      row_topghost[i] = row_bottom[i];
      row_bottomghost[i] = row_top[i];
    }

    // Left & right wrap
    for (int j = 1; j <= y_size; j++) {
      double *row = new_points + j * f_xsize;
      row[0] = row[x_size];     // left ghost = right boundary
      row[x_size + 1] = row[1]; // right ghost = left boundary
    }
  }
}

/*
 * NOTE: this routine a good candidate for openmp
 *       parallelization
 */
inline void get_total_energy(const uint size[2], const double *plane,
                             double *energy) {

  register const int x_size = size[_x_];
  register const int y_size = size[_y_];

#if defined(LONG_ACCURACY)
  long double tot_energy = 0;
#else
  double tot_energy = 0;
#endif

// HACK :Review this code snippet
// HINT: you may attempt to
//       (i)  manually unroll the loop
//       (ii) ask the compiler to do it
// for instance
#pragma omp parallel for reduction(+ : tot_energy)
  for (uint j = 1; j <= y_size; ++j) {
    const double *row = plane + j * (size[_x_] + 2);
    // Automatically hints the compiler to do simd reduction here
#pragma omp simd reduction(+ : tot_energy)
    for (uint i = 1; i <= x_size; ++i)
      tot_energy += row[i];
  }

  *energy = (double)tot_energy;
}
