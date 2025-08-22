/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

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

typedef uint vec2_t[2];
typedef double *restrict buffers_t[4];

typedef struct {
  // restrict makes sure that the pointers are the only references
  // to the memory they point to
  double *restrict data;
  vec2_t size;
} plane_t;

extern int inject_energy(const int, const int, const vec2_t *, const double,
                         plane_t *, const vec2_t);

extern int send_halos(buffers_t *, vec2_t, uint *, MPI_Comm *);
extern int recv_halos(buffers_t *, vec2_t, uint *, MPI_Comm *);

extern int update_plane(const int, const vec2_t, const plane_t *, plane_t *);

extern int get_total_energy(plane_t *, double *);

int initialize(MPI_Comm *, int, int, int, char **, vec2_t *, vec2_t *, int *,
               int *, uint *, int *, int *, int *, vec2_t **, double *,
               plane_t *, buffers_t *);

int memory_release(plane_t *, buffers_t *);

int output_energy_stat(int, plane_t *, double, int, MPI_Comm *);

inline int inject_energy(const int periodic, const int Nsources,
                         const vec2_t *Sources, const double energy,
                         plane_t *plane, const vec2_t N) {
  const uint register sizex = plane->size[_x_] + 2;
  double *restrict data = plane->data;

#define IDX(i, j) ((j) * sizex + (i))
  for (int s = 0; s < Nsources; s++) {
    int x = Sources[s][_x_];
    int y = Sources[s][_y_];

    data[IDX(x, y)] += energy;

    if (periodic) {
      if (x == 1)
        plane->data[IDX(N[_x_] + 1, y)] += energy;
      if ((N[_x_] == 1)) {
        plane->data[IDX(0, y)] += energy;
      }
      if (y == 1)
        plane->data[IDX(x, N[_y_] + 1)] += energy;
      if ((N[_y_] == 1)) {
        plane->data[IDX(x, 0)] += energy;
      }
    }
  }
#undef IDX

  return 0;
}

// NOTE: Inline the function for efficiency
inline extern int send_halos(buffers_t *buffers, vec2_t size, uint *neighbours,
                             MPI_Comm *Comm) {

  MPI_Send(buffers[SEND][NORTH], size[_x_], MPI_DOUBLE, neighbours[NORTH], SEND,
           *Comm);
  MPI_Send(buffers[SEND][SOUTH], size[_x_], MPI_DOUBLE, neighbours[SOUTH], SEND,
           *Comm);
  MPI_Send(buffers[SEND][EAST], size[_y_], MPI_DOUBLE, neighbours[EAST], SEND,
           *Comm);
  MPI_Send(buffers[SEND][WEST], size[_y_], MPI_DOUBLE, neighbours[WEST], SEND,
           *Comm);
  return 0;
}

inline extern int recv_halos(buffers_t *buffers, vec2_t size, uint *neighbours,
                             MPI_Comm *Comm) {

  // TODO: Complete this
  MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
           MPI_Comm comm, MPI_Status *status);

  return 0;
}

inline int update_plane(const int periodic,
                        const vec2_t N, // the grid of MPI tasks
                        const plane_t *oldplane, plane_t *newplane)

{
  uint register fxsize = oldplane->size[_x_] + 2;
  uint register fysize = oldplane->size[_y_] + 2;

  uint register xsize = oldplane->size[_x_];
  uint register ysize = oldplane->size[_y_];

#define IDX(i, j) ((j) * fxsize + (i))

  double *restrict old = oldplane->data;
  double *restrict new = newplane->data;

  for (uint j = 1; j <= ysize; j++)
    for (uint i = 1; i <= xsize; i++) {

      // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
      //       if this patch is at some border without periodic conditions;
      //       in that case it is assumed that the +-1 points are outside the
      //       plate and always have a value of 0, i.e. they are an
      //       "infinite sink" of heat
      //
      // NOTE: That if here I put an if statement (for example to check the
      // borders) it is likely that the compiler will not perform vectorization
      // by himself automatically

      // five-points stencil formula
      //
      // HINT : check the serial version for some optimization
      //
      new[IDX(i, j)] =
          old[IDX(i, j)] / 2.0 + (old[IDX(i - 1, j)] + old[IDX(i + 1, j)] +
                                  old[IDX(i, j - 1)] + old[IDX(i, j + 1)]) /
                                     4.0 / 2.0;
    }

  // TODO: Check if here I can simply take the code from the serial version
  // Perhaps I need to adjust things to work for the patches, though each plane
  // will have the corresponding size which helps identify and are different for
  // different ranks if I understood correctly
  if (periodic) {
    if (N[_x_] == 1) {
      // propagate the boundaries as needed
      // check the serial version
    }

    if (N[_y_] == 1) {
      // propagate the boundaries as needed
      // check the serial version
    }
  }

#undef IDX
  return 0;
}

inline int get_total_energy(plane_t *plane, double *energy) {

  const int register x_size = plane->size[_x_];
  const int register y_size = plane->size[_y_];
  const int register f_size = x_size + 2;

#if defined(LONG_ACCURACY)
  long double totenergy = 0;
#else
  double totenergy = 0;
#endif

#pragma omp parallel for reduction(+ : totenergy)
  for (uint j = 1; j <= y_size; ++j) {
    const double *row = plane->data + j * f_size;
    // Automatically hints the compiler to do simd reduction here
#pragma omp simd reduction(+ : totenergy)
    for (uint i = 1; i <= x_size; ++i)
      totenergy += row[i];
  }

  *energy = (double)totenergy;
  return 0;
}

// WARNING: Old inefficient version
// inline int get_total_energy(plane_t *plane, double *energy) {
//
//   const int register xsize = plane->size[_x_];
//   const int register ysize = plane->size[_y_];
//   const int register fsize = xsize + 2;
//
//   double *restrict data = plane->data;
//
// #define IDX(i, j) ((j) * fsize + (i))
//
// #if defined(LONG_ACCURACY)
//   long double totenergy = 0;
// #else
//   double totenergy = 0;
// #endif
//
//   for (int j = 1; j <= ysize; j++)
//     for (int i = 1; i <= xsize; i++)
//       totenergy += data[IDX(i, j)];
//
// #undef IDX
//
//   *energy = (double)totenergy;
//   return 0;
// }
