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

#define MEMORY_ALIGNMENT 64 // 64 bytes = 512 bits alignment for SIMD

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
extern void initialize_send_buffers_datatype(buffers_t buffers[2], plane_t *);

extern error_code_t exchange_halos(buffers_t buffers[2], vec2_t size,
                                   int *neighbours, MPI_Comm *Comm,
                                   MPI_Request requests[8]);

extern error_code_t exchange_halos_datatypes(buffers_t buffers[2],
                                             int *neighbours, MPI_Comm *Comm,
                                             MPI_Request requests[8],
                                             MPI_Datatype north_south_type,
                                             MPI_Datatype east_west_type);

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

inline void initialize_send_buffers_datatype(buffers_t buffers[2],
                                             plane_t *plane) {
  const uint size_x = plane->size[_x_];
  const uint size_y = plane->size[_y_];
  const uint stride = size_x + 2;

  // Starting from the internal row so I have to move by 1 stride + 1
  // NORTH send buffer points to the first element of the top internal row
  buffers[SEND][NORTH] = &plane->data[stride + 1];
  // Fill SOUTH buffer points to the first bottom internal row
  buffers[SEND][SOUTH] = &plane->data[size_y * stride + 1];
  // Start position of the WEST buffer leftmost internal column
  buffers[SEND][WEST] = &plane->data[stride + 1];
  // Start position of the EAST buffer rightmost internal column
  buffers[SEND][EAST] = &plane->data[stride + size_x];

  // ghost row 0 first element
  buffers[RECV][NORTH] = &plane->data[1];
  // ghost row at the bottom
  buffers[RECV][SOUTH] = &plane->data[(size_y + 1) * stride + 1];

  // ghost row 1 first (halo element)
  buffers[RECV][WEST] = &plane->data[stride];
  // ghost row at east (halo element)
  buffers[RECV][EAST] = &plane->data[stride + size_x + 1];
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

error_code_t exchange_halos(buffers_t buffers[2], vec2_t size, int *neighbours,
                            MPI_Comm *Comm, MPI_Request requests[8]) {
  int rc = MPI_SUCCESS; // Accumulate MPI errors with OR to check all at once

  // NOTE: Keep the receive first send after order we would use for blocking
  // version which avoid deadlocks

  // NORTH-SOUTH exchanges, also making an or with the return value for later
  rc |= MPI_Irecv(buffers[RECV][SOUTH], size[_x_], MPI_DOUBLE,
                  neighbours[SOUTH], NORTH, *Comm, &requests[0]);
  rc |= MPI_Isend(buffers[SEND][NORTH], size[_x_], MPI_DOUBLE,
                  neighbours[NORTH], NORTH, *Comm, &requests[1]);

  rc |= MPI_Irecv(buffers[RECV][NORTH], size[_x_], MPI_DOUBLE,
                  neighbours[NORTH], SOUTH, *Comm, &requests[2]);
  rc |= MPI_Isend(buffers[SEND][SOUTH], size[_x_], MPI_DOUBLE,
                  neighbours[SOUTH], SOUTH, *Comm, &requests[3]);

  // EAST-WEST exchanges
  rc |= MPI_Irecv(buffers[RECV][WEST], size[_y_], MPI_DOUBLE, neighbours[WEST],
                  EAST, *Comm, &requests[4]);
  rc |= MPI_Isend(buffers[SEND][EAST], size[_y_], MPI_DOUBLE, neighbours[EAST],
                  EAST, *Comm, &requests[5]);

  rc |= MPI_Irecv(buffers[RECV][EAST], size[_y_], MPI_DOUBLE, neighbours[EAST],
                  WEST, *Comm, &requests[6]);
  rc |= MPI_Isend(buffers[SEND][WEST], size[_y_], MPI_DOUBLE, neighbours[WEST],
                  WEST, *Comm, &requests[7]);

  // Single check at the end
  if (rc != MPI_SUCCESS) {
    return ERROR_MPI_FAILURE;
  }

  return SUCCESS;
}

extern error_code_t exchange_halos_datatypes(buffers_t buffers[2],
                                             int *neighbours, MPI_Comm *Comm,
                                             MPI_Request requests[8],
                                             MPI_Datatype north_south_type,
                                             MPI_Datatype east_west_type) {
  int rc = MPI_SUCCESS; // Accumulate MPI errors with OR to check all at once

  // NORTH-SOUTH exchanges, also making an or with the return value for later
  // NOTE: count = 1 because we send the entire pre-committed datatype
  // NOTE: MPI automatically handles sends/receives to MPI_PROC_NULL as no-ops
  // NOTE: Keep the receive first send after order we would use for blocking
  // version which avoid deadlocks
  rc |= MPI_Irecv(buffers[RECV][SOUTH], 1, north_south_type, neighbours[SOUTH],
                  NORTH, *Comm, &requests[0]);
  rc |= MPI_Isend(buffers[SEND][NORTH], 1, north_south_type, neighbours[NORTH],
                  NORTH, *Comm, &requests[1]);

  rc |= MPI_Irecv(buffers[RECV][NORTH], 1, north_south_type, neighbours[NORTH],
                  SOUTH, *Comm, &requests[2]);
  rc |= MPI_Isend(buffers[SEND][SOUTH], 1, north_south_type, neighbours[SOUTH],
                  SOUTH, *Comm, &requests[3]);

  // // EAST-WEST exchanges
  rc |= MPI_Irecv(buffers[RECV][WEST], 1, east_west_type, neighbours[WEST],
                  EAST, *Comm, &requests[4]);
  rc |= MPI_Isend(buffers[SEND][EAST], 1, east_west_type, neighbours[EAST],
                  EAST, *Comm, &requests[5]);

  rc |= MPI_Irecv(buffers[RECV][EAST], 1, east_west_type, neighbours[EAST],
                  WEST, *Comm, &requests[6]);
  rc |= MPI_Isend(buffers[SEND][WEST], 1, east_west_type, neighbours[WEST],
                  WEST, *Comm, &requests[7]);

  // Single check at the end
  if (rc != MPI_SUCCESS) {
    return ERROR_MPI_FAILURE;
  }

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
