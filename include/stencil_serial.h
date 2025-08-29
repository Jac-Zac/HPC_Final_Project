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

#define NORTH 0
#define SOUTH 1
#define EAST 2
#define WEST 3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

// ============================================================
//
// function prototypes

int initialize(int, char **, uint *, int *, int *, int *, int **, double *,
               double **, int *);

int memory_release(double *, int *);

extern int inject_energy(const int, const int, const int *, const double,
                         const uint[2], double *);

extern int update_plane(const int, const uint[2], const double *, double *);

#ifdef GCC_EXTENSIONS
typedef double v2df __attribute__((vector_size(2 * sizeof(double))));
#endif

extern int get_total_energy(const uint[2], const double *, double *);

// ============================================================
//
// function definition for inline functions

inline int inject_energy(const int periodic, const int Nsources,
                         const int *Sources, const double energy,
                         const uint mysize[2], double *plane) {
#define IDX(i, j) ((j) * (mysize[_x_] + 2) + (i))
  for (int s = 0; s < Nsources; s++) {

    uint x = Sources[2 * s];
    uint y = Sources[2 * s + 1];
    plane[IDX(x, y)] += energy;

    if (periodic) {
      if (x == 1)
        plane[IDX(mysize[_x_] + 1, y)] += energy;
      if (x == mysize[_x_])
        plane[IDX(0, y)] += energy;
      if (y == 1)
        plane[IDX(x, mysize[_y_] + 1)] += energy;
      if (y == mysize[_y_])
        plane[IDX(x, 0)] += energy;
    }
  }
#undef IDX

  return 0;
}

#define PREFETCHING
#ifndef PREFETCHING
/*
 * calculate the new energy values
 * the old plane contains the current data, the new plane
 * will store the updated data
 *
 * NOTE: in parallel, every MPI task will perform the
 *       calculation for its patch
 */
inline int update_plane(const int periodic, const uint size[2],
                        const double *old_points, double *new_points) {

  // clang: ISO C++17 does not allow 'register' storage class specifier
  const int f_xsize = size[_x_] + 2;
  const int x_size = size[_x_];
  const int y_size = size[_y_];

  // Pre-compute constants to avoid repeated calculations
  const double alpha = 0.6;
  const double beta = (1.0 - alpha) * 0.25; // (1-alpha)/4

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

  return 0;
}

#else

// A macro for finding the minimum of two values, common in pure C
#define min(a, b) (((a) < (b)) ? (a) : (b))

// A reasonable block size. This is a crucial tuning parameter!
// A size of 32x32 doubles = 32 * 32 * 8 bytes = 8KB,
// which fits comfortably in a typical 32KB L1 data cache.
#define BLOCK_SIZE 16

inline int update_plane(const int periodic, const uint size[2],
                        const double *old_points, double *new_points) {
  const int f_xsize = size[_x_] + 2;
  const int x_size = size[_x_];
  const int y_size = size[_y_];

  const double alpha = 0.6;
  const double beta = (1.0 - alpha) * 0.25;

  // 1. Calculate the boundaries for the main work (full-sized tiles)
  // This finds the largest coordinate that is a multiple of BLOCK_SIZE.
  const int y_main_end = (y_size / BLOCK_SIZE) * BLOCK_SIZE;
  const int x_main_end = (x_size / BLOCK_SIZE) * BLOCK_SIZE;

  // =======================================================================
  // 2. FAST PATH: Process the main body of full BLOCK_SIZE x BLOCK_SIZE tiles
  // In this section, there are NO `min()` calls and NO `cmp` instructions
  // for calculating loop bounds. The compiler knows the exact loop trip count.
  // =======================================================================
  for (int jj = 1; jj <= y_main_end; jj += BLOCK_SIZE) {
    for (int ii = 1; ii <= x_main_end; ii += BLOCK_SIZE) {
      // Process one full tile without any boundary checks
      for (int j = jj; j < jj + BLOCK_SIZE; ++j) {
        const double *row_above = old_points + (j - 1) * f_xsize;
        const double *row_center = old_points + j * f_xsize;
        const double *row_below = old_points + (j + 1) * f_xsize;
        double *row_new = new_points + j * f_xsize;
        for (int i = ii; i < ii + BLOCK_SIZE; ++i) {
          const double center = row_center[i];
          const double left = row_center[i - 1];
          const double right = row_center[i + 1];
          const double up = row_above[i];
          const double down = row_below[i];
          row_new[i] = center * alpha + beta * (left + right + up + down);
        }
      }
    }
  }

  // =======================================================================
  // 3. CLEANUP: Handle the remaining "fringe" tiles on the right and bottom
  // This code is similar to the original but only runs for the edge cases.
  // =======================================================================
  // Process remaining tiles in the bottom fringe (all columns)
  for (int jj = y_main_end + 1; jj <= y_size; jj += BLOCK_SIZE) {
    for (int ii = 1; ii <= x_size; ii += BLOCK_SIZE) {
      const int j_end = min(jj + BLOCK_SIZE, y_size + 1);
      const int i_end = min(ii + BLOCK_SIZE, x_size + 1);
      for (int j = jj; j < j_end; ++j) {
        const double *row_above = old_points + (j - 1) * f_xsize;
        const double *row_center = old_points + j * f_xsize;
        const double *row_below = old_points + (j + 1) * f_xsize;
        double *row_new = new_points + j * f_xsize;
        for (int i = ii; i < i_end; ++i) {
          row_new[i] = old_points[j * f_xsize + i] * alpha +
                       beta * (old_points[j * f_xsize + i - 1] +
                               old_points[j * f_xsize + i + 1] +
                               old_points[(j - 1) * f_xsize + i] +
                               old_points[(j + 1) * f_xsize + i]);
        }
      }
    }
  }

  // Process remaining tiles in the right fringe (only for the main body rows)
  for (int jj = 1; jj <= y_main_end; jj += BLOCK_SIZE) {
    for (int ii = x_main_end + 1; ii <= x_size; ii += BLOCK_SIZE) {
      const int j_end = min(jj + BLOCK_SIZE, y_size + 1);
      const int i_end = min(ii + BLOCK_SIZE, x_size + 1);
      for (int j = jj; j < j_end; ++j) {
        const double *row_above = old_points + (j - 1) * f_xsize;
        const double *row_center = old_points + j * f_xsize;
        const double *row_below = old_points + (j + 1) * f_xsize;
        double *row_new = new_points + j * f_xsize;
        for (int i = ii; i < i_end; ++i) {
          row_new[i] = old_points[j * f_xsize + i] * alpha +
                       beta * (old_points[j * f_xsize + i - 1] +
                               old_points[j * f_xsize + i + 1] +
                               old_points[(j - 1) * f_xsize + i] +
                               old_points[(j + 1) * f_xsize + i]);
        }
      }
    }
  }

  /* Periodic boundary update logic remains the same */
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

  return 0;
}

// Don't forget to undefine the macro if it's in a header file
// to avoid potential conflicts with other code.
#undef min

#endif

/*
 * NOTE: this routine a good candidate for openmp
 *       parallelization
 */
inline int get_total_energy(const uint size[2], const double *plane,
                            double *energy) {

  register const int x_size = size[_x_];
  register const int y_size = size[_y_];

#if defined(LONG_ACCURACY)
  long double totenergy = 0;
#else
  double totenergy = 0;
#endif

// HACK :Review this code snippet
// HINT: you may attempt to
//       (i)  manually unroll the loop
//       (ii) ask the compiler to do it
// for instance
#pragma omp parallel for reduction(+ : totenergy)
  for (uint j = 1; j <= y_size; ++j) {
    const double *row = plane + j * (size[_x_] + 2);
    // Automatically hints the compiler to do simd reduction here
#pragma omp simd reduction(+ : totenergy)
    for (uint i = 1; i <= x_size; ++i)
      totenergy += row[i];
  }

  *energy = (double)totenergy;
  return 0;
}
