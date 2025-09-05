#include "stencil_parallel.h"

int main(int argc, char **argv) {
  MPI_Comm my_COMM_WORLD;
  int rank, num_tasks;
  int neighbours[4];

  int num_iterations;
  int periodic;
  vec2_t global_grid_size, mpi_grid_dims;

  int num_sources;
  int num_sources_local;
  vec2_t *sources_local;
  double energy_per_source;

  plane_t planes[2];
  buffers_t buffers[2];

  int output_energy_stats;

  // initialize MPI envrionment
  {
    int level_obtained;

    // TODO: change MPI_FUNNELED if appropriate
    // NOTE: MPI_THREAD_MULTIPLE: Multiple threads may call MPI at any time
    // without restrictions. (Might be useful)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    if (level_obtained < MPI_THREAD_FUNNELED) {
      fprintf(
          stderr,
          "Rank %d: ERROR - MPI thread level obtained (%d) is insufficient. "
          "Required: %d (MPI_THREAD_FUNNELED)\n",
          rank, level_obtained, MPI_THREAD_FUNNELED);
      MPI_Finalize();
      exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &my_COMM_WORLD);
  }

  // argument checking and setting
  int ret =
      initialize(&my_COMM_WORLD, rank, num_tasks, argc, argv, &global_grid_size,
                 &mpi_grid_dims, &periodic, &output_energy_stats, neighbours,
                 &num_iterations, &num_sources, &num_sources_local,
                 &sources_local, &energy_per_source, &planes[0], &buffers[0]);

  if (ret) {
    printf("task %d is opting out with termination code %d\n", rank, ret);

    MPI_Finalize();
    return SUCCESS;
  }

  // Allocate timing arrays for logging
  double *comp_times = (double *)malloc(num_iterations * sizeof(double));
  double *comm_times = (double *)malloc(num_iterations * sizeof(double));
  if (comp_times == NULL || comm_times == NULL) {
    fprintf(stderr,
            "Rank %d: ERROR - Failed to allocate timing arrays "
            "(comp_times: %p, comm_times: %p)\n",
            rank, (void *)comp_times, (void *)comm_times);
    MPI_Abort(my_COMM_WORLD, 1);
  }

  // Synchronize all ranks before starting the timer for better reproducibility
  MPI_Barrier(my_COMM_WORLD);

  double total_start_time = MPI_Wtime();
  int current = OLD;
  for (int iter = 0; iter < num_iterations; ++iter) {
    double t_comm_start, t_comp_start;

    MPI_Request reqs[8];

    // new energy from sources
    inject_energy(periodic, num_sources_local, sources_local, energy_per_source,
                  &planes[current], mpi_grid_dims);

    /* --------------------------------------------------------------------- */

    // [A] fill the buffers, and/or make the buffers' pointers pointing to the
    // correct position
    //
    // Extract current boundary data into send buffers
    //

    /* --- COMMUNICATION PHASE --- */
    t_comm_start = MPI_Wtime();
    // Pass periodic and neighbouur if they are null pass nothing
    fill_send_buffers(buffers, &planes[current]);

    // [B] perform the halo communications
    //     (1) use Send / Recv
    //     (2) use Isend / Irecv
    //         --> can you overlap communication and companion in this way?
    //
    // Send new halos the one that was just computed
    //
    // Initialize array of statuses
    MPI_Status statuses[4];

    int ret = exchange_halos(buffers, planes[current].size, neighbours,
                             &my_COMM_WORLD, statuses);

    // Return if unsuccessful
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "Rank %d: MPI halo exchange failed with error %d\n", rank,
              ret);
      MPI_Abort(my_COMM_WORLD, ret);
      return ERROR_MPI_FAILURE;
    }

    // [C] copy the haloes data
    copy_received_halos(buffers, &planes[current], neighbours);

    comm_times[iter] = MPI_Wtime() - t_comm_start;
    /* --------------------------------------  */

    /* --- COMPUTATION PHASE --- */
    t_comp_start = MPI_Wtime();

    // update grid points
    update_plane(periodic, mpi_grid_dims, &planes[current], &planes[!current]);

    comp_times[iter] = MPI_Wtime() - t_comp_start;

    /* ------------------------- */
    // output if needed
    if (output_energy_stats) {
      output_energy_stat(iter, &planes[!current],
                         (iter + 1) * num_sources * energy_per_source, rank,
                         &my_COMM_WORLD);

      char filename[128];
      sprintf(filename, "data_logging/%d_plane_%05d.bin", rank, iter);
      int dump_status =
          dump(planes[!current].data, planes[!current].size, filename);
      if (dump_status != 0) {
        fprintf(stderr, "Error in dump_status. Exit with %d\n", dump_status);
      }

      // // swap plane indexes for the new iteration
      current = !current;
    }
  }

  // Synchronize before final timing
  MPI_Barrier(my_COMM_WORLD);
  double total_time = MPI_Wtime() - total_start_time;

  double total_comp_time_local = 0.0;
  double total_comm_time_local = 0.0;
  for (int i = 0; i < num_iterations; ++i) {
    total_comp_time_local += comp_times[i];
    total_comm_time_local += comm_times[i];
  }

  double max_comp_time, max_comm_time;

  // Reduce with MPI_MAX get the maximum computation time from all processes,
  // storing the result in max_comp_time on the root process (rank 0).
  MPI_Reduce(&total_comp_time_local, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             my_COMM_WORLD);
  MPI_Reduce(&total_comm_time_local, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             my_COMM_WORLD);

  double t_tot_energy_start = MPI_Wtime();
  output_energy_stat(-1, &planes[!current],
                     num_iterations * num_sources * energy_per_source, rank,
                     &my_COMM_WORLD);
  double t_tot_energy_end = MPI_Wtime();

  if (rank == 0) {
    printf("Total time: %f seconds\n", total_time);
    printf("Max computation time across ranks: %f seconds\n", max_comp_time);
    printf("Max communication time across ranks: %f seconds\n", max_comm_time);
    printf("Total energy computaton time: %f \n",
           t_tot_energy_end - t_tot_energy_start);
  }

  // --- Cleanup ---
  if (comp_times != NULL) {
    free(comp_times);
  }
  if (comm_times != NULL) {
    free(comm_times);
  }
  memory_release(planes, buffers);

  MPI_Finalize();
  return SUCCESS;
}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ==========================================================================
 */

/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ==========================================================================
 */

uint simple_factorization(uint, int *, uint **);

int initialize_sources(int, int, MPI_Comm *, uint[2], int, int *, vec2_t **,
                       int);

int memory_allocate(const int *, const vec2_t *, buffers_t *, plane_t *);

error_code_t
initialize(MPI_Comm *Comm, // the communicator
           int Me,         // the rank of the calling process
           int Ntasks,     // the total number of MPI ranks
           int argc,       // the argc from command line
           char **argv,    // the argv from command line
           vec2_t *S,      // the size of the plane
           vec2_t *N,      // two-uint array defining the MPI tasks' grid
           int *periodic,  // periodic-boundary tag
           int *output_energy_stat,
           int *neighbours,  // four-int array that gives back the
                             // neighbours rank of the calling task
           int *Niterations, // how many iterations
           int *Nsources,    // how many heat sources
           int *Nsources_local, vec2_t **Sources_local,
           double *energy_per_source, // how much heat per source
           plane_t *planes, buffers_t *buffers) {
  int halt = 0;
  int ret;
  int verbose = 0;
  int testing = 0;

  // ··································································
  // set fixed values for testing

  (*S)[_x_] = 10000;
  (*S)[_y_] = 10000;
  *periodic = 0;
  *Nsources = 4;
  *Nsources_local = 0;
  *Sources_local = NULL;
  *Niterations = 1000;
  *energy_per_source = 1.0;
  *output_energy_stat = 0;

  if (planes == NULL) {
    // Just on prints the error
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - planes pointer is NULL\n", Me);
    return ERROR_NULL_POINTER;
  }

  // NOTE: What was that
  planes[OLD].size[0] = 0;
  planes[NEW].size[0] = 0;

  // Set the neighbours to MPI null as default
  for (int i = 0; i < 4; i++)
    neighbours[i] = MPI_PROC_NULL;

  // Set initially the buffers to null pointres
  for (int b = 0; b < 2; b++)
    for (int d = 0; d < 4; d++)
      buffers[b][d] = NULL;

  // ··································································
  // process the command line
  while (1) {
    int opt;
    while ((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:t:")) != -1) {
      switch (opt) {
      case 'x':
        (*S)[_x_] = (uint)atoi(optarg);
        break;

      case 'y':
        (*S)[_y_] = (uint)atoi(optarg);
        break;

      case 'e':
        *Nsources = atoi(optarg);
        break;

      case 'E':
        *energy_per_source = atof(optarg);
        break;

      case 'n':
        *Niterations = atoi(optarg);
        break;

      case 'o':
        *output_energy_stat = (atoi(optarg) > 0);
        break;

      case 'p':
        *periodic = (atoi(optarg) > 0);
        break;

      case 'v':
        verbose = atoi(optarg);
        break;
      case 't':
        testing = atol(optarg);
        break;
      case 'h': {
        if (Me == 0)
          printf("\nValid options (values in [] are defaults):\n"
                 "-x    x size of the plate [10000]\n"
                 "-y    y size of the plate [10000]\n"
                 "-e    number of energy sources [4]\n"
                 "-E    energy per source [1.0]\n"
                 "-n    number of iterations [1000]\n"
                 "-o    output energy stats [0 = disabled, 1 = enabled]\n"
                 "-t    testing mode [0 = disabled, 1 = enabled]\n"
                 "-p    periodic boundaries [0 = false, 1 = true]\n"
                 "-v    verbosity level\n"
                 "-h    show this help message\n\n");
        halt = 1;
      } break;

      case ':':
        printf("option -%c requires an argument\n", optopt);
        break;

      case '?':
        printf(" -------- help unavailable ----------\n");
        break;
      }
    }

    if (opt == -1)
      break;
  }

  if (halt)
    return SUCCESS;

  // ··································································
  // Comprehensive input validation
  if ((*S)[_x_] <= 0 || (*S)[_y_] <= 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Grid dimensions must be positive, got %d x %d\n",
              (*S)[_x_], (*S)[_y_]);
    return ERROR_INVALID_GRID_SIZE;
  }
  if (*Nsources < 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Number of sources must be >= 0, got %d\n",
              *Nsources);
    return ERROR_INVALID_NUM_SOURCES;
  }
  if (*Niterations <= 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Number of iterations must be > 0, got %d\n",
              *Niterations);
    return ERROR_INVALID_NUM_ITERATIONS;
  }
  if (*energy_per_source <= 0.0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Energy per source must be > 0.0, got %f\n",
              *energy_per_source);
    return ERROR_INVALID_ENERGY_VALUE;
  }

  // ··································································
  /*
   * find a suitable domain decomposition
   * very simple algorithm, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   */

  vec2_t Grid;
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_] / (*S)[_y_]
                                              : (double)(*S)[_y_] / (*S)[_x_]);
  int dimensions = 2 - (Ntasks <= ((int)formfactor + 1));

  if (dimensions == 1) {
    if ((*S)[_x_] >= (*S)[_y_])
      Grid[_x_] = Ntasks, Grid[_y_] = 1;
    else
      Grid[_x_] = 1, Grid[_y_] = Ntasks;
  } else {
    int nf;
    uint *factors;
    uint first = 1;
    ret = simple_factorization(Ntasks, &nf, &factors);

    // for (int i = 0; (i < nf) && ((Ntasks / first) / first > formfactor);
    // i++) NOTE: Adding explicit casting
    for (int i = 0; i < nf && ((double)Ntasks / (first * first) > formfactor);
         i++)
      first *= factors[i];

    if ((*S)[_x_] > (*S)[_y_])
      Grid[_x_] = Ntasks / first, Grid[_y_] = first;
    else
      Grid[_x_] = first, Grid[_y_] = Ntasks / first;
  }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];

  // ··································································
  // my coordinates in the grid of processors based on my rank
  uint X = Me % Grid[_x_];
  uint Y = Me / Grid[_x_];

  // ··································································
  // find my neighbours
  if (Grid[_x_] > 1) {
    if (*periodic) {
      neighbours[EAST] = Y * Grid[_x_] + (Me + 1) % Grid[_x_];
      // NOTE: Old version without casting to int
      // neighbours[WEST] = (X % Grid[_x_] > 0 ? Me - 1 : (Y + 1) * Grid[_x_] -
      // 1);
      neighbours[WEST] =
          (X % Grid[_x_] > 0 ? (int)(Me - 1) : (int)((Y + 1) * Grid[_x_] - 1));
    }

    else {
      neighbours[EAST] = (X < Grid[_x_] - 1 ? Me + 1 : MPI_PROC_NULL);
      neighbours[WEST] = (X > 0 ? (Me - 1) % Ntasks : MPI_PROC_NULL);
    }
  }

  if (Grid[_y_] > 1) {
    if (*periodic) {
      neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
      neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks;
    }

    else {
      // NOTE: Old version
      // neighbours[NORTH] = (Y > 0 ? Me - Grid[_x_] : MPI_PROC_NULL);
      // neighbours[SOUTH] = (Y < Grid[_y_] - 1 ? Me + Grid[_x_] :
      // MPI_PROC_NULL);
      //
      neighbours[NORTH] = (Y > 0 ? (int)(Me - Grid[_x_]) : MPI_PROC_NULL);
      neighbours[SOUTH] =
          (Y < Grid[_y_] - 1 ? (int)(Me + Grid[_x_]) : MPI_PROC_NULL);
    }
  }

  // ··································································
  // the size of my patch

  /*
   * every MPI task determines the size sx x sy of its own domain
   * REMIND: the computational domain will be embedded into a frame
   *         that is (sx+2) x (sy+2)
   *         the outer frame will be used for halo communication or
   */

  vec2_t my_size;

  // Split in a balanced way
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];

  // Give extra cells to the first r tasks
  my_size[_x_] = s + (X < r);

  // Do the same for the y axies
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  my_size[_y_] = s + (Y < r);

  planes[OLD].size[0] = my_size[0];
  planes[OLD].size[1] = my_size[1];
  planes[NEW].size[0] = my_size[0];
  planes[NEW].size[1] = my_size[1];

  if (verbose > 0) {
    if (Me == 0) {
      printf("Tasks are decomposed in a grid %d x %d\n\n", Grid[_x_],
             Grid[_y_]);
      fflush(stdout);
    }

    MPI_Barrier(*Comm);

    for (int t = 0; t < Ntasks; t++) {
      if (t == Me) {
        printf("Task %4d :: "
               "\tgrid coordinates : %3d, %3d\n"
               "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
               Me, X, Y, neighbours[NORTH], neighbours[EAST], neighbours[SOUTH],
               neighbours[WEST]);
        fflush(stdout);
      }

      MPI_Barrier(*Comm);
    }
  }

  // ··································································
  // allocate the needed memory
  ret = memory_allocate(neighbours, N, buffers, planes);

  if (ret != 0) {
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - Memory allocation failed\n", Me);
    return ret;
  }

  // ··································································
  // heat sources are local to the specific patch (thus to the specific rank)
  ret = initialize_sources(Me, Ntasks, Comm, my_size, *Nsources, Nsources_local,
                           Sources_local, testing);
  if (ret != 0) {
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - Failed to initialize sources\n", Me);
    return ERROR_INITIALIZE_SOURCES;
  }

  return SUCCESS;
}

// NOTE: Think of a better factorization which has more square like patches to
// have better cache locality for OpenMP threads computations
//
// Think of using a Cartesian decomposition Topology-aware decomposition: With
// the correct topology:
// https://edoras.sdsu.edu/~mthomas/sp17.605/lectures/MPI-Cart-Comms-and-Topos.pdf
uint simple_factorization(uint a, int *nfactors, uint **factors)
/*
 * rough factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  int n = 0;
  uint f = 2;
  uint _a_ = a;

  while (f < a) {
    while (_a_ % f == 0) {
      n++;
      _a_ /= f;
    }
    f++;
  }

  *nfactors = n;
  uint *_factors_ = (uint *)malloc(n * sizeof(uint));

  n = 0;
  f = 2;
  _a_ = a;

  while (f < a) {
    while (_a_ % f == 0) {
      _factors_[n++] = f;
      _a_ /= f;
    }
    f++;
  }

  *factors = _factors_;
  return 0;
}

// NOTE: To review with previous implementation
int initialize_sources(int Me, int Ntasks, MPI_Comm *Comm, vec2_t mysize,
                       int Nsources, int *Nsources_local, vec2_t **sources,
                       int testing) {
  // Use deterministic seed when testing for reproducible results
  if (testing != 0) {
    srand48(1337 ^ Me); // Fixed seed for testing
  } else {
    srand48(time(NULL) ^ Me);
  }
  int *tasks_with_sources = (int *)malloc(Nsources * sizeof(int));

  if (Me == 0) {
    for (int i = 0; i < Nsources; i++)
      tasks_with_sources[i] = (int)lrand48() % Ntasks;
  }

  MPI_Bcast(tasks_with_sources, Nsources, MPI_INT, 0, *Comm);

  int nlocal = 0;
  for (int i = 0; i < Nsources; i++)
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;

  if (nlocal > 0) {
    vec2_t *restrict helper = (vec2_t *)malloc(nlocal * sizeof(vec2_t));
    for (int s = 0; s < nlocal; s++) {
      helper[s][_x_] = 1 + lrand48() % mysize[_x_];
      helper[s][_y_] = 1 + lrand48() % mysize[_y_];
    }

    *sources = helper;

    if (testing != 0) {

      char fname[256];
      snprintf(fname, sizeof(fname), "data_logging/sources_rank%d.txt", Me);
      FILE *f = fopen(fname, "w");
      if (f) {
        for (int s = 0; s < nlocal; s++) {
          fprintf(f, "%d %d\n", helper[s][_x_], helper[s][_y_]);
        }
        fclose(f);
      } else {
        fprintf(stderr, "Rank %d: Failed to open file %s for writing\n", Me,
                fname);
      }
    }
  }

  free(tasks_with_sources);

  return SUCCESS;
}

// NOTE: In the future I have to think carefully about the fact that if I want
// to parallelize inside those patches the allocation should be done by the
// threads to have a touch first policy perhaps
int memory_allocate(const int *neighbours, const vec2_t *N,
                    buffers_t *buffers_ptr, plane_t *planes_ptr) {
  /*
    here you allocate the memory buffers that you need to
    (i)  hold the results of your computation
    (ii) communicate with your neighbours

    The memory layout that I propose to you is as follows:

    (i) --- calculations
    you need 2 memory regions: the "OLD" one that contains the
    results for the step (i-1)th, and the "NEW" one that will contain
    the updated results from the step ith.

    Then, the "NEW" will be treated as "OLD" and viceversa.

    These two memory regions are indexed by *plate_ptr:

    planew_ptr[0] ==> the "OLD" region
    plamew_ptr[1] ==> the "NEW" region


    (ii) --- communications

    You may need two buffers (one for sending and one for receiving)
    for each one of your neighbours, that are at most 4:
    north, south, east and west.

    To them you need to communicate at most my_size_x or my_size_y
    double data.

    These buffers are indexed by the buffer_ptr pointer so
    that

    (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
    (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions

    --->> Of course you can change this layout as you prefer

   */

  if (planes_ptr == NULL)
    return ERROR_NULL_POINTER;
  if (buffers_ptr == NULL)
    return ERROR_NULL_POINTER;

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size =
      (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2);

  // HACK: Testing memory alignment to get aligned SIMD instructions
  if (posix_memalign((void **)&planes_ptr[OLD].data, MEMORY_ALIGNMENT,
                     frame_size * sizeof(double)) != 0) {
    return ERROR_MEMORY_ALLOCATION;
  }

  if (posix_memalign((void **)&planes_ptr[NEW].data, MEMORY_ALIGNMENT,
                     frame_size * sizeof(double)) != 0) {
    free(planes_ptr[OLD].data);
    return ERROR_MEMORY_ALLOCATION;
  }

  // OLD: Standard malloc allocation (commented out but preserved)
  // planes_ptr[OLD].data = (double *)malloc(frame_size * sizeof(double));
  // if (planes_ptr[OLD].data == NULL)
  //   // manage the malloc fail
  //   return ERROR_MEMORY_ALLOCATION;
  //
  // planes_ptr[NEW].data = (double *)malloc(frame_size * sizeof(double));
  // if (planes_ptr[NEW].data == NULL)
  //   // manage the malloc fail
  //   return ERROR_MEMORY_ALLOCATION;

  // NOTE: This old allocation method doesn't touch memory correctly
  // memset(planes_ptr[OLD].data, 0, frame_size * sizeof(double));
  // memset(planes_ptr[NEW].data, 0, frame_size * sizeof(double));

  // Initialize memory by touching it correctly
  const uint f_xsize = planes_ptr->size[_x_] + 2;
  const uint xsize = planes_ptr->size[_x_];
  const uint ysize = planes_ptr->size[_y_];

#pragma omp parallel for schedule(static)
  for (uint j = 0; j < ysize + 2; ++j) {
    for (uint i = 0; i < xsize + 2; ++i) {
      planes_ptr[OLD].data[j * f_xsize + i] = 0.0;
      planes_ptr[NEW].data[j * f_xsize + i] = 0.0;
    }
  }

  // ··················································
  // NOTE: This comment is done by the professor
  // buffers for north and south communication
  // are not really needed
  //
  // in fact, they are already contiguous, just the
  // first and last line of every rank's plane
  //
  // you may just make some pointers pointing to the
  // correct positions
  // TODO: Complete the following code
  uint size_x = planes_ptr[OLD].size[_x_];
  uint size_y = planes_ptr[OLD].size[_y_];

  // +1 to skip the halo since we want the actual data I believe
  buffers_ptr[SEND][NORTH] = &planes_ptr[OLD].data[1 * (size_x + 2) + 1];
  buffers_ptr[SEND][SOUTH] = &planes_ptr[OLD].data[size_y * (size_x + 2) + 1];

  // NOTE: The rest of the buffers so the RECV for north and south are set to
  // NULL before so no need to do it here

  // NOTE: Do not do this ! I think it is not optimal !
  //
  // or, if you prefer, just go on and allocate buffers
  // also for north and south communications

  // ··················································
  // allocate buffers
  // Both send and rexieve for west and east
  buffers_ptr[SEND][WEST] =
      (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
  buffers_ptr[RECV][WEST] =
      (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  buffers_ptr[SEND][EAST] =
      (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
  buffers_ptr[RECV][EAST] =
      (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  // ··················································

  return SUCCESS;
}

// Release memory also for the buffers
error_code_t memory_release(plane_t *planes, buffers_t *buffer_ptr) {
  if (planes != NULL) {
    if (planes[OLD].data != NULL)
      free(planes[OLD].data);

    if (planes[NEW].data != NULL)
      free(planes[NEW].data);
  }

  // Free only EAST and WEST buffers (NORTH/SOUTH point to plane data)
  // RECV buffers
  if (buffer_ptr[RECV][WEST] != NULL) {
    free(buffer_ptr[RECV][WEST]);
  }

  if (buffer_ptr[RECV][EAST] != NULL) {
    free(buffer_ptr[RECV][EAST]);
  }

  // SEND buffers
  if (buffer_ptr[SEND][WEST] != NULL) {
    free(buffer_ptr[SEND][WEST]);
  }
  if (buffer_ptr[SEND][EAST] != NULL) {
    free(buffer_ptr[SEND][EAST]);
  }

  return SUCCESS;
}

error_code_t output_energy_stat(int step, plane_t *plane, double budget, int Me,
                                MPI_Comm *Comm) {
  // Set initial energy
  double system_energy = 0;
  double tot_system_energy = 0;
  // Every rank compute its total energy for the patch using parallel reduction
  get_total_energy(plane, &system_energy);

  // Reduce by patch to get the final energy
  // NOTE: To review actually
  MPI_Reduce(&system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0,
             *Comm);

  if (Me == 0) {
    if (step >= 0)
      printf(" [ step %4d ] ", step);
    fflush(stdout);

    printf("total injected energy is %g, "
           "system energy is %g "
           "( in avg %g per grid point)\n",
           budget, tot_system_energy,
           tot_system_energy / (plane->size[_x_] * plane->size[_y_]));
  }

  return SUCCESS;
}

error_code_t dump(const double *data, const uint size[2],
                  const char *filename) {
  // Function to dump each rank ptach into a file
  if ((filename != NULL) && (filename[0] != '\0')) {
    FILE *outfile = fopen(filename, "wb");
    if (outfile == NULL)
      return ERROR_MEMORY_ALLOCATION;

    double *array = (double *)malloc(size[0] * sizeof(double));

    for (uint j = 1; j <= size[1]; j++) {
      const double *restrict line = data + j * (size[0] + 2);
      for (uint i = 1; i <= size[0]; i++) {
        array[i - 1] = (double)line[i];
      }
      fwrite(array, sizeof(double), size[0], outfile);
    }

    free(array);

    fclose(outfile);
    return SUCCESS;
  } else
    return ERROR_NULL_POINTER;
}
