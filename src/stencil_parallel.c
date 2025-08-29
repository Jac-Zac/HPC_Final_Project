/*
 *
 *  my_size_x   :   local x-extension of your patch
 *  my_size_y   :   local y-extension of your patch
 *
 */

#include "stencil_parallel.h"

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv) {
  MPI_Comm my_COMM_WORLD;
  int Rank, Ntasks;
  int neighbours[4];

  int Niterations;
  int periodic;
  vec2_t S, N;

  int Nsources;
  int Nsources_local;
  vec2_t *Sources_local;
  double energy_per_source;

  plane_t planes[2];
  buffers_t buffers[2];

  int output_energy_stat_per_step;

  // initialize MPI envrionment
  {
    int level_obtained;

    // TODO: change MPI_FUNNELED if appropriate
    // NOTE: MPI_THREAD_MULTIPLE: Multiple threads may call MPI at any time
    // without restrictions. (Might be useful)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    if (level_obtained < MPI_THREAD_FUNNELED) {
      printf("MPI_thread level obtained is %d instead of %d\n", level_obtained,
             MPI_THREAD_FUNNELED);
      MPI_Finalize();
      exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &my_COMM_WORLD);
  }

  // argument checking and setting
  int ret = initialize(&my_COMM_WORLD, Rank, Ntasks, argc, argv, &S, &N,
                       &periodic, &output_energy_stat_per_step, neighbours,
                       &Niterations, &Nsources, &Nsources_local, &Sources_local,
                       &energy_per_source, &planes[0], &buffers[0]);

  if (ret) {
    printf("task %d is opting out with termination code %d\n", Rank, ret);

    MPI_Finalize();
    return 0;
  }

  // Allocate timing arrays for logging
  double *comp_times = (double *)malloc(Niterations * sizeof(double));
  double *comm_times = (double *)malloc(Niterations * sizeof(double));
  if (comp_times == NULL || comm_times == NULL) {
    fprintf(stderr, "Rank %d: Failed to allocate timing arrays.\n", Rank);
    MPI_Abort(my_COMM_WORLD, 1);
  }

  // Synchronize all ranks before starting the timer for better reproducibility
  MPI_Barrier(my_COMM_WORLD);

  double total_start_time = MPI_Wtime();
  int current = OLD;
  for (int iter = 0; iter < Niterations; ++iter) {
    double t_comm_start, t_comp_start;

    // MPI_Request reqs[8];

    // new energy from sources
    inject_energy(periodic, Nsources_local, Sources_local, energy_per_source,
                  &planes[current], N);

    /* --------------------------------------------------------------------- */

    // [A] fill the buffers, and/or make the buffers' pointers pointing to the
    // correct position
    //
    // Extract current boundary data into send buffers
    //

    /* --- COMMUNICATION PHASE --- */
    t_comm_start = MPI_Wtime();
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

    exchange_halos(buffers, planes[current].size, neighbours, &my_COMM_WORLD,
                   statuses);

    // [C] copy the haloes data
    copy_received_halos(buffers, &planes[current], neighbours);

    comm_times[iter] = MPI_Wtime() - t_comm_start;
    /* --------------------------------------  */

    /* --- COMPUTATION PHASE --- */
    t_comp_start = MPI_Wtime();

    // update grid points
    // update_plane(periodic, N, &planes[current], &planes[!current]);
    // update_plane_parallel(periodic, N, &planes[current], &planes[!current]);
    update_plane_parallel_tiling(periodic, N, &planes[current],
                                 &planes[!current]);

    comp_times[iter] = MPI_Wtime() - t_comp_start;

    /* ------------------------- */
    // output if needed
    if (output_energy_stat_per_step) {
      output_energy_stat(iter, &planes[!current],
                         (iter + 1) * Nsources * energy_per_source, Rank,
                         &my_COMM_WORLD);
      // char filename[128];
      // sprintf(filename, "plane_global_%05d.bin", iter);
      //
      // Call dump_global_grid
      // dump_global_grid(&planes[!current],     // local plane
      //                  planes[!current].size, // my patch size
      //                  S,                     // global size of grid
      //                  coords,        // coordinates of this rank in MPI grid
      //                  my_COMM_WORLD, // MPI communicator
      //                  filename);

      // swap plane indexes for the new iteration
      current = !current;
    }
  }

  // Synchronize before final timing
  MPI_Barrier(my_COMM_WORLD);
  double total_time = MPI_Wtime() - total_start_time;

  double total_comp_time_local = 0.0;
  double total_comm_time_local = 0.0;
  for (int i = 0; i < Niterations; ++i) {
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
                     Niterations * Nsources * energy_per_source, Rank,
                     &my_COMM_WORLD);
  double t_tot_energy_end = MPI_Wtime();

  if (Rank == 0) {
    printf("Total time: %f seconds\n", total_time);
    printf("Max computation time across ranks: %f seconds\n", max_comp_time);
    printf("Max communication time across ranks: %f seconds\n", max_comm_time);
    printf("Total energy computaton time: %f \n",
           t_tot_energy_end - t_tot_energy_start);
  }

  // --- Cleanup ---
  free(comp_times);
  free(comm_times);
  memory_release(planes, buffers);

  MPI_Finalize();
  return 0;
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

int initialize(MPI_Comm *Comm,
               int Me,        // the rank of the calling process
               int Ntasks,    // the total number of MPI ranks
               int argc,      // the argc from command line
               char **argv,   // the argv from command line
               vec2_t *S,     // the size of the plane
               vec2_t *N,     // two-uint array defining the MPI tasks' grid
               int *periodic, // periodic-boundary tag
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
  int seed = -1;

  // ··································································
  // set default values

  (*S)[_x_] = 10000;
  (*S)[_y_] = 10000;
  *periodic = 0;
  *Nsources = 4;
  *Nsources_local = 0;
  *Sources_local = NULL;
  *Niterations = 1000;
  *energy_per_source = 1.0;

  if (planes == NULL) {
    // Just on prints the error
    if (Me == 0)
      fprintf(stderr, "Error: planes pointer is NULL\n");
    return 5; // error code
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
    while ((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:s:")) != -1) {
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
      case 's':
        seed = atol(optarg);
        break;
      case 'h': {
        if (Me == 0)
          printf("\nvalid options are ( values btw [] are the default values "
                 "):\n"
                 "-x    x size of the plate [10000]\n"
                 "-y    y size of the plate [10000]\n"
                 "-e    how many energy sources on the plate [4]\n"
                 "-E    how many energy sources on the plate [1.0]\n"
                 "-n    how many iterations [1000]\n"
                 "-p    whether periodic boundaries applies  [0 = false]\n\n");
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
    return 1;

  // ··································································
  // TODO: Complete checks for meaningful values

  if ((*S)[_x_] <= 0 || (*S)[_y_] <= 0) {
    if (Me == 0)
      fprintf(stderr, "Error: grid size must be positive\n");
    return 2;
  }
  if (*Nsources < 0) {
    if (Me == 0)
      fprintf(stderr, "Error: Nsources must be >= 0\n");
    return 3;
  }
  if (*Niterations <= 0) {
    if (Me == 0)
      fprintf(stderr, "Error: Niterations must be > 0\n");
    return 4;
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
    int Nf;
    uint *factors;
    uint first = 1;
    ret = simple_factorization(Ntasks, &Nf, &factors);

    // for (int i = 0; (i < Nf) && ((Ntasks / first) / first > formfactor);
    // i++) NOTE: Adding explicit casting
    for (int i = 0; i < Nf && ((double)Ntasks / (first * first) > formfactor);
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
      // NOTE: Old version
      // neighbours[WEST] = (X % Grid[_x_] > 0 ? Me - 1 : (Y + 1) * Grid[_x_]
      // - 1);
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

  // NOTE: Give extra cells to the first r tasks
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

  // ··································································
  // heat sources are local to the specific patch (thus to the specific rank)
  ret = initialize_sources(Me, Ntasks, Comm, my_size, *Nsources, Nsources_local,
                           Sources_local, seed);
  if (ret != 0) {
    if (Me == 0)
      fprintf(stderr, "Error initializing sources\n");
    return 6;
  }

  return 0;
}

// NOTE: Think of a better factorization which has more square like patches to
// have better cache locality for OpenMP threads computations
//
// Think of using a Cartesian decomposition Topology-aware decomposition: With
// the correct topology:
// https://edoras.sdsu.edu/~mthomas/sp17.605/lectures/MPI-Cart-Comms-and-Topos.pdf
uint simple_factorization(uint A, int *Nfactors, uint **factors)
/*
 * rough factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  int N = 0;
  uint f = 2;
  uint _A_ = A;

  while (f < A) {
    while (_A_ % f == 0) {
      N++;
      _A_ /= f;
    }
    f++;
  }

  *Nfactors = N;
  uint *_factors_ = (uint *)malloc(N * sizeof(uint));

  N = 0;
  f = 2;
  _A_ = A;

  while (f < A) {
    while (_A_ % f == 0) {
      _factors_[N++] = f;
      _A_ /= f;
    }
    f++;
  }

  *factors = _factors_;
  return 0;
}

int initialize_sources(int Me, int Ntasks, MPI_Comm *Comm, vec2_t mysize,
                       int Nsources, int *Nsources_local, vec2_t **Sources,
                       int seed)

{
  if (seed < 0) {
    // Do not set a custom seed if not defined
    srand48(time(NULL) ^ Me);
  } else {
    // Set a fixed seed if one is defined
    srand48(seed ^ Me);
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

    *Sources = helper;
  }

#ifdef PRINT_SOURCES
  if (*Nsources_local > 0) {
    for (int s = 0; s < *Nsources_local; s++) {
      printf("Rank %d source %d: (%d, %d)\n", Me, s, (*Sources)[s][_x_],
             (*Sources)[s][_y_]);
    }
  }
  MPI_Barrier(*Comm); // synchronize for cleaner output
#endif

  free(tasks_with_sources);

  return 0;
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
    // an invalid pointer has been passed
    // manage the situation
    return 1;

  if (buffers_ptr == NULL)
    // an invalid pointer has been passed
    // manage the situation
    return 2;

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size =
      (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2);

  planes_ptr[OLD].data = (double *)malloc(frame_size * sizeof(double));
  if (planes_ptr[OLD].data == NULL)
    // manage the malloc fail
    return -1;
  memset(planes_ptr[OLD].data, 0, frame_size * sizeof(double));

  planes_ptr[NEW].data = (double *)malloc(frame_size * sizeof(double));
  if (planes_ptr[NEW].data == NULL)
    // manage the malloc fail
    return -2;
  memset(planes_ptr[NEW].data, 0, frame_size * sizeof(double));

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
  buffers_ptr[SEND][WEST] = malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
  buffers_ptr[RECV][WEST] = malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  buffers_ptr[SEND][EAST] = malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
  buffers_ptr[RECV][EAST] = malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  // ··················································

  return 0;
}

// Release memory also for the buffers
int memory_release(plane_t *planes, buffers_t *buffer_ptr) {
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

  return 0;
}

// NOTE: Review this code though It is all written by the professor
int output_energy_stat(int step, plane_t *plane, double budget, int Me,
                       MPI_Comm *Comm) {
  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy(plane, &system_energy);

  // Reduce by patch knowing system energy
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

  return 0;
}

// NOTE: TO FIX
// dump global grid
// ...
