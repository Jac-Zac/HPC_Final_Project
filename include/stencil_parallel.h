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

// #define MEMORY_ALIGNMENT 64  // 64 bytes = 512 bits alignment for SIMD
#define MEMORY_ALIGNMENT 32 // 64 bytes = 512 bits alignment for SIMD

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
  ERROR_FILE_DUMPING = 8
} error_code_t;

typedef uint vec2_t[2];
typedef double *restrict buffers_t[4];
// NOTE: Added a more specific definition
typedef struct {
  // restrict makes sure that the pointers are the only references
  // to the memory they point to
  double *restrict data;
  vec2_t size;
} plane_t;

extern void inject_energy(const int, const int, const vec2_t *, const double,
                          plane_t *, const vec2_t);

extern void fill_send_buffers(buffers_t buffers[2], plane_t *);

// extern void send_halos(buffers_t *, vec2_t, int *, MPI_Comm *);
// extern void recv_halos(buffers_t *, vec2_t, int *, MPI_Comm *,
//                        MPI_Status statuses[4]);

extern error_code_t exchange_halos(buffers_t buffers[2], vec2_t size,
                                   int *neighbours, MPI_Comm *Comm,
                                   MPI_Status statuses[4]);

extern void copy_received_halos(buffers_t buffers[2], plane_t *, int *);

extern void update_plane(const int, const vec2_t, const plane_t *, plane_t *);

extern void get_total_energy(plane_t *, double *);

error_code_t initialize(MPI_Comm *, int, int, int, char **, vec2_t *, vec2_t *,
                        int *, int *, int *, int *, int *, int *, vec2_t **,
                        double *, plane_t *, buffers_t *);

error_code_t memory_release(plane_t *, buffers_t *);

error_code_t output_energy_stat(int, plane_t *, double, int, MPI_Comm *);

extern error_code_t dump(const double *data, const uint size[2],
                         const char *filename);

inline void inject_energy(const int periodic, const int Nsources,
                          const vec2_t *Sources, const double energy,
                          plane_t *plane, const vec2_t N) {
  register const uint size_x = plane->size[_x_] + 2;

  double *restrict data = plane->data;

#define IDX(i, j) ((j) * size_x + (i))
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
}

void fill_send_buffers(buffers_t buffers[2], plane_t *plane) {
  const uint size_x = plane->size[_x_];
  const uint size_y = plane->size[_y_];
  const uint stride = size_x + 2;

  // NOTE: For north and south we can copy continues memory directly starting
  // from the correct location

  // Starting from the internal row so I have to move by 1 stride + 1
  // Fill NORTH send buffer (top internal row, j=1)
  memcpy(buffers[SEND][NORTH], &plane->data[1 * stride + 1],
         size_x * sizeof(double));

  // Fill SOUTH send buffer (bottom internal row, j=size_y)
  memcpy(buffers[SEND][SOUTH], &plane->data[size_y * stride + 1],
         size_x * sizeof(double));

  // ghost row 0
  buffers[RECV][NORTH] = &plane->data[0 * stride + 1];
  // ghost row sy+1
  buffers[RECV][SOUTH] = &plane->data[(size_y + 1) * stride + 1];

  // NOTE: Perhaps here I should do it by thread so that they touch first the
  // memory location
  //
  // Fill WEST send buffer (leftmost internal column, i=1)
  for (uint j = 0; j < size_y; j++) {
    buffers[SEND][WEST][j] = plane->data[(j + 1) * stride + 1];
  }

  // Fill EAST send buffer (rightmost internal column, i=size_x)
  for (uint j = 0; j < size_y; j++) {
    buffers[SEND][EAST][j] = plane->data[(j + 1) * stride + size_x];
  }
}

void copy_received_halos(buffers_t buffers[2], plane_t *plane,
                         int *neighbours) {
  const uint size_y = plane->size[_y_];
  const uint size_x = plane->size[_x_];
  const uint stride = size_x + 2;

  // NORTH and SOUTH are already in place thanks to direct MPI_Recv.
  // We only need to unpack the non-contiguous columns.

  // Copy WEST halo (from EAST neighbor)
  if (neighbours[WEST] != MPI_PROC_NULL) {
    for (uint j = 0; j < size_y; j++) {
      plane->data[(j + 1) * stride + 0] = buffers[RECV][WEST][j];
    }
  }

  // Copy EAST halo (from WEST neighbor)
  if (neighbours[EAST] != MPI_PROC_NULL) {
    for (uint j = 0; j < size_y; j++) {
      plane->data[(j + 1) * stride + size_x + 1] = buffers[RECV][EAST][j];
    }
  }
}

// extern void send_halos(buffers_t *buffers, vec2_t size, int *neighbours,
//                        MPI_Comm *Comm) {
//
//   // Neighbours tells me which rank to send to and the tag tells what I'm
//   // sending to which neighbour, thous he will know what it is receiving
//
//   // Get num points based on the size of x since those are horizontal
//   MPI_Send((const void *)buffers[SEND][NORTH], size[_x_], MPI_DOUBLE,
//            neighbours[NORTH], NORTH, *Comm);
//
//   MPI_Send((const void *)buffers[SEND][SOUTH], size[_x_], MPI_DOUBLE,
//            neighbours[SOUTH], SOUTH, *Comm);
//
//   MPI_Send((const void *)buffers[SEND][EAST], size[_y_], MPI_DOUBLE,
//            neighbours[EAST], EAST, *Comm);
//   MPI_Send((const void *)buffers[SEND][WEST], size[_y_], MPI_DOUBLE,
//            neighbours[WEST], WEST, *Comm);
// }
//
// extern void recv_halos(buffers_t *buffers, vec2_t size, int *neighbours,
//                        MPI_Comm *Comm, MPI_Status statuses[4]) {
//
//   // NOTE: When I receive from north I should put it in the
//   // south since the neighbour is the north neighbour.
//   MPI_Recv((void *)buffers[RECV][NORTH], size[_x_], MPI_DOUBLE,
//            neighbours[SOUTH], SOUTH, *Comm, &statuses[0]);
//   MPI_Recv((void *)buffers[RECV][SOUTH], size[_x_], MPI_DOUBLE,
//            neighbours[NORTH], NORTH, *Comm, &statuses[1]);
//   MPI_Recv((void *)buffers[RECV][WEST], size[_y_], MPI_DOUBLE,
//   neighbours[EAST],
//            EAST, *Comm, &statuses[2]);
//   MPI_Recv((void *)buffers[RECV][EAST], size[_y_], MPI_DOUBLE,
//   neighbours[WEST],
//            WEST, *Comm, &statuses[3]);
// }

// NOTE: Tmp version to test things out
error_code_t exchange_halos(buffers_t buffers[2], vec2_t size, int *neighbours,
                            MPI_Comm *Comm, MPI_Status statuses[4]) {
  int rc;

  rc = MPI_Sendrecv(buffers[SEND][NORTH], size[_x_], MPI_DOUBLE,
                    neighbours[NORTH], SOUTH, buffers[RECV][SOUTH], size[_x_],
                    MPI_DOUBLE, neighbours[SOUTH], SOUTH, *Comm, &statuses[0]);

  if (rc != MPI_SUCCESS)
    return ERROR_MPI_FAILURE;

  rc = MPI_Sendrecv(buffers[SEND][SOUTH], size[_x_], MPI_DOUBLE,
                    neighbours[SOUTH], NORTH, buffers[RECV][NORTH], size[_x_],
                    MPI_DOUBLE, neighbours[NORTH], NORTH, *Comm, &statuses[1]);

  if (rc != MPI_SUCCESS)
    return ERROR_MPI_FAILURE;

  rc = MPI_Sendrecv(buffers[SEND][EAST], size[_y_], MPI_DOUBLE,
                    neighbours[EAST], WEST, buffers[RECV][WEST], size[_y_],
                    MPI_DOUBLE, neighbours[WEST], WEST, *Comm, &statuses[2]);

  if (rc != MPI_SUCCESS)
    return ERROR_MPI_FAILURE;

  rc = MPI_Sendrecv(buffers[SEND][WEST], size[_y_], MPI_DOUBLE,
                    neighbours[WEST], EAST, buffers[RECV][EAST], size[_y_],
                    MPI_DOUBLE, neighbours[EAST], EAST, *Comm, &statuses[3]);

  if (rc != MPI_SUCCESS)
    return ERROR_MPI_FAILURE;

  return SUCCESS;
}

inline void update_plane(const int periodic,
                         const vec2_t N, // MPI grid of ranks
                         const plane_t *oldplane, plane_t *newplane) {

  const uint f_xsize = oldplane->size[_x_] + 2;
  const uint xsize = oldplane->size[_x_];
  const uint ysize = oldplane->size[_y_];

  double *restrict newp = newplane->data;
  const double *restrict oldp = oldplane->data;

  // Use defined stencil coefficients
  const double c_center = 0.5;
  const double c_neigh = 0.125; // 1/8

// Row-parallel, inner loop vectorized by compiler
#pragma omp parallel for schedule(static)
  for (uint j = 1; j <= ysize; ++j) {
    const double *row_above = oldp + (j - 1) * f_xsize;
    const double *row_center = oldp + j * f_xsize;
    const double *row_below = oldp + (j + 1) * f_xsize;
    double *row_new = newp + j * f_xsize;

#pragma omp simd
    for (uint i = 1; i <= xsize; ++i) {

      // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
      //       if this patch is at some border without periodic conditions;
      //       in that case it is assumed that the +-1 points are outside the
      //       plate and always have a value of 0, i.e. they are an
      //       "infinite sink" of heat
      //
      // NOTE: That if here I put an if statement (for example to check the
      // borders) it is likely that the compiler will not perform
      // vectorization by himself automatically
      const double center = row_center[i];
      const double left = row_center[i - 1];
      const double right = row_center[i + 1];
      const double up = row_above[i];
      const double down = row_below[i];

      row_new[i] = center * c_center + (left + right + up + down) * c_neigh;
    }
  }

  // Periodic propagation for single-rank-in-dimension cases (your original
  // intent)
  if (periodic) {
    // If only one rank along X, wrap left/right ghosts locally
    if (N[_x_] == 1) {
      for (uint j = 1; j <= ysize; ++j) {
        double *row = newp + j * f_xsize;
        row[0] = row[xsize];     // left ghost  <= right edge
        row[xsize + 1] = row[1]; // right ghost <= left  edge
      }
    }
    // If only one rank along Y, wrap top/bottom ghosts locally
    if (N[_y_] == 1) {
      double *row_top = newp + 1 * f_xsize;
      double *row_bottom = newp + ysize * f_xsize;
      double *row_topghost = newp + 0 * f_xsize;
      double *row_bottomghost = newp + (ysize + 1) * f_xsize;

      for (uint i = 1; i <= xsize; ++i) {
        row_topghost[i] = row_bottom[i];
        row_bottomghost[i] = row_top[i];
      }
    }
  }

  // // TODO: Check if here I can simply take the code from the serial version
  // // Perhaps I need to adjust things to work for the patches, though each
  // // plane will have the corresponding size which helps identify and are
  // // different for different ranks if I understood correctly
  // if (periodic) {
  //   if (N[_x_] == 1) {
  //     // propagate the boundaries as needed
  //     // check the serial version
  //   }
  //
  //   if (N[_y_] == 1) {
  //     // propagate the boundaries as needed
  //     // check the serial version
  //   }
  // }
}

inline void get_total_energy(plane_t *plane, double *energy) {

  register const int x_size = plane->size[_x_];
  register const int y_size = plane->size[_y_];
  register const int f_size = x_size + 2;

  double *restrict data = plane->data;

#if defined(LONG_ACCURACY)
  long double totenergy = 0;
#else
  double totenergy = 0;
#endif

#pragma omp parallel for reduction(+ : totenergy)
  for (uint j = 1; j <= y_size; ++j) {
    const double *row = data + j * f_size;
    // Automatically hints the compiler to do simd reduction here
#pragma omp simd reduction(+ : totenergy)
    for (uint i = 1; i <= x_size; ++i)
      totenergy += row[i];
  }

  *energy = (double)totenergy;
}
