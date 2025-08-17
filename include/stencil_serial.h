/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <float.h>
#include <getopt.h>
#include <math.h>
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
               double **, int *, int *);

int memory_release(double *, int *);

extern int inject_energy(const int, const int, const int *, const double,
                         const uint[2], double *);

extern int update_plane(const int, const uint[2], const double *, double *);

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

// HACK: This is actually faster
inline int update_plane_fast(const int periodic, const uint size[2],
                             const double *old_points, double *new_points) {

  const int fxsize = size[_x_] + 2;
  const int fysize = size[_y_] + 2;
  const int xsize = size[_x_];
  const int ysize = size[_y_];

  const double alpha = 0.6;
  const double beta = (1.0 - alpha) * 0.25; // (1-alpha)/4

#define IDX(i, j) ((j) * fxsize + (i))

  // Parallelize outer loop over rows: each thread works on full rows (avoids
  // false sharing)
#pragma omp parallel for schedule(static)
  for (int j = 1; j <= ysize; j++) {
    const double *row_above = old_points + (j - 1) * fxsize;
    const double *row_center = old_points + j * fxsize;
    const double *row_below = old_points + (j + 1) * fxsize;
    double *row_new = new_points + j * fxsize;

    for (int i = 1; i <= xsize; i++) {
      // Five-point stencil
      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];

      row_new[i] = center * alpha + beta * (left + right + up + down);
    }
  }

  // Periodic boundary propagation
  if (periodic) {
    // Top & bottom wrap
    for (int i = 1; i <= xsize; i++) {
      new_points[i] = new_points[IDX(i, ysize)];
      new_points[IDX(i, ysize + 1)] = new_points[i];
    }
    // Left & right wrap
    for (int j = 1; j <= ysize; j++) {
      new_points[IDX(0, j)] = new_points[IDX(xsize, j)];
      new_points[IDX(xsize + 1, j)] = new_points[IDX(1, j)];
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
inline int update_plane(const int periodic, const uint size[2],
                        const double *old_points, double *new_points) {

  // clang: ISO C++17 does not allow 'register' storage class specifier
  const int fxsize = size[_x_] + 2;
  const int fysize = size[_y_] + 2;
  const int xsize = size[_x_];
  const int ysize = size[_y_];

  // Pre-compute constants to avoid repeated calculations
  const double alpha = 0.6;
  const double beta = (1.0 - alpha) * 0.25; // (1-alpha)/4

#define IDX(i, j) ((j) * fxsize + (i))
#define OLD_VERSION 0

// HINT: you may attempt to
//       (i)  manually unroll the loop
//       (ii) ask the compiler to do it
// for instance
// #pragma GCC unroll 4
// NOTE: loop unrolling doesn't seem to increase performance
//
// HINT: in any case, this loop is a good candidate
//       for openmp parallelization
//
// #pragma omp parallel for collapse(2)
// #pragma omp parallel for schedule(dynamic)
#pragma omp parallel for schedule(static)
  for (int j = 1; j <= ysize; j++) {
    for (int i = 1; i <= xsize; i++) {
#if OLD_VERSION == 1
      //
      // five-points stencil formula
      //

      // simpler stencil with no explicit diffusivity
      // always conserve the smoothed quantity
      // alpha here mimics how much "easily" the heat travels
      double result = old_points[IDX(i, j)] * alpha;
      double sum_i = (old_points[IDX(i - 1, j)] + old_points[IDX(i + 1, j)]) /
                     4.0 * (1 - alpha);
      double sum_j = (old_points[IDX(i, j - 1)] + old_points[IDX(i, j + 1)]) /
                     4.0 * (1 - alpha);
      result += (sum_i + sum_j);
      new_points[IDX(i, j)] = result;
#else
      // Cache array accesses and use pointer arithmetic for better performance
      // NOTE: Computation of the index is not efficient perhaps but I guess the
      // reason because collapse is not faster is indeed that the compiler
      // automatically does hoist j * fxsize outside the i loop
      const double center = old_points[IDX(i, j)];
      const double left = old_points[IDX(i - 1, j)];
      const double right = old_points[IDX(i + 1, j)];
      const double up = old_points[IDX(i, j - 1)];
      const double down = old_points[IDX(i, j + 1)];

      // Optimized computation - fewer operations
      new_points[IDX(i, j)] =
          center * alpha + beta * (left + right + up + down);
#endif

      // NOTE: Professor left this comment I'm not really sure what I have to do
      // with it if it should be implemented or anything
      /*
      // implantation from the derivation of
      // 3-points 2nd order derivatives
      // however, that should depends on an adaptive
      // time-stepping so that given a diffusivity
      // coefficient the amount of energy diffused is
      // "small"
      // however the implicit methods are not stable

  #define alpha_guess 0.5 // mimic the heat diffusivity

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
  }

  /*
   * propagate boundaries if they are periodic
   *
   * NOTE: when is that needed in distributed memory, if any?
   */
  if (periodic) {
    for (int i = 1; i <= xsize; i++) {
      new_points[i] = new_points[IDX(i, ysize)];
      new_points[IDX(i, ysize + 1)] = new_points[i];
    }
    for (int j = 1; j <= ysize; j++) {
      new_points[IDX(0, j)] = new_points[IDX(xsize, j)];
      new_points[IDX(xsize + 1, j)] = new_points[IDX(1, j)];
    }
  }

  return 0;

#undef IDX
}

/*
 * NOTE: this routine a good candidate for openmp
 *       parallelization
 */
inline int get_total_energy(const uint size[2], const double *plane,
                            double *energy) {

  // clang: ISO C++17 does not allow 'register' storage class specifier
  const int xsize = size[_x_];

#define IDX(i, j) ((j) * (xsize + 2) + (i))

#if defined(LONG_ACCURACY)
  long double totenergy = 0;
#else
  double totenergy = 0;
#endif

  // HINT: you may attempt to
  //       (i)  manually unroll the loop
  //       (ii) ask the compiler to do it
  // for instance
  // #pragma omp parallel for
  //
#pragma omp parallel for reduction(+ : totenergy)
  for (uint j = 1; j <= size[_y_]; j++)
    for (uint i = 1; i <= size[_x_]; i++)
      totenergy += plane[IDX(i, j)];

#undef IDX

  *energy = (double)totenergy;
  return 0;
}
