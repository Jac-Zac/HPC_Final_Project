#include "stencil_parallel.h"
#include "mpi.h"

int main(int argc, char **argv) {
  MPI_Comm my_COMM_WORLD;
  int rank, num_tasks;
  int neighbours[4];

  int num_iterations;
  int periodic;
  vec2_t global_grid_size, mpi_tasks_grid;

  int num_sources;
  int num_sources_local;
  vec2_t *sources_local;
  double energy_per_source;

  plane_t planes[2];

  int output_energy_stats;

  // initialize MPI envrionment
  {
    int level_obtained;

    // Use MPI_THREAD_FUNNELED: Only main thread makes MPI calls
    // This is appropriate for OpenMP+MPI hybrid where OpenMP threads don't call MPI
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
                 &mpi_tasks_grid, &periodic, &output_energy_stats, neighbours,
                 &num_iterations, &num_sources, &num_sources_local,
                 &sources_local, &energy_per_source, &planes[0]);

  if (ret) {
    if (ret != HELP_DISPLAYED) {
      printf("task %d is opting out with termination code %d\n", rank, ret);
    }
    
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

  // Define and commit custom MPI datatypes for efficient halo exchange
  // Using MPI datatypes avoids explicit buffer copies and leverages MPI's
  // optimized data movement for non-contiguous memory regions
  int x_size = planes[OLD].size[_x_];
  int y_size = planes[OLD].size[_y_];
  MPI_Datatype north_south_type;
  MPI_Datatype east_west_type;

  // North-South exchanges: data is contiguous in memory (full rows)
  MPI_Type_contiguous(x_size, MPI_DOUBLE, &north_south_type);

  // East-West exchanges: data is non-contiguous (single column with stride)
  // Use vector datatype to handle strided access pattern
  int element_per_block = 1; // One element per block (single column)
  int stride = x_size + 2;   // Total row width including halos
  // https: // rookiehpc.org/mpi/docs/mpi_type_vector/index.html
  MPI_Type_vector(y_size, element_per_block, stride, MPI_DOUBLE,
                  &east_west_type);

  ret = MPI_Type_commit(&north_south_type);
  if (ret != MPI_SUCCESS)
    return ret;

  ret = MPI_Type_commit(&east_west_type);
  if (ret != MPI_SUCCESS)
    return ret;

  int current = OLD;
  for (int iter = 0; iter < num_iterations; ++iter) {
    double t_comm_start, t_comp_start;

    MPI_Request requests[8];

    // new energy from sources
    inject_energy(periodic, num_sources_local, sources_local, energy_per_source,
                  &planes[current], mpi_tasks_grid);

    /* --------------------------------------------------------------------- */

    /* --- COMMUNICATION PHASE --- */
    t_comm_start = MPI_Wtime();
    error_code_t ret =
        exchange_halos(&planes[current], neighbours, &my_COMM_WORLD, requests,
                       north_south_type, east_west_type);

    // MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    // Return if unsuccessful
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "Rank %d: MPI halo exchange failed with error %d\n", rank,
              ret);
      MPI_Abort(my_COMM_WORLD, ret);
      return ERROR_MPI_FAILURE;
    }
    comm_times[iter] = MPI_Wtime() - t_comm_start;
    /* --------------------------------------  */

    /* --- COMPUTATION PHASE --- */
    t_comp_start = MPI_Wtime();

    // update grid points
    // update_plane(periodic, mpi_tasks_grid, &planes[current],
    // &planes[!current]); update grid points
    update_plane_inner(&planes[current], &planes[!current]);

    comp_times[iter] = MPI_Wtime() - t_comp_start;
    /* ------------------------- */

    /* --- ADDITIONAL COMMUNICATION PHASE --- */
    t_comm_start = MPI_Wtime();
    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
    comm_times[iter] += MPI_Wtime() - t_comm_start;
    /* --------------------------------------  */

    /* --- COMPUTATION PHASE --- */
    t_comp_start = MPI_Wtime();
    update_plane_borders(periodic, mpi_tasks_grid, &planes[current],
                         &planes[!current]);
    comp_times[iter] += MPI_Wtime() - t_comp_start;
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
    }

    // Swap plane roles: NEW becomes OLD for next iteration
    current = !current;
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
  memory_release(planes, &north_south_type, &east_west_type);

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

int memory_allocate(plane_t *);

error_code_t initialize(
    MPI_Comm *Comm,         // the communicator
    int Me,                 // the rank of the calling process
    int num_tasks,          // the total number of MPI ranks
    int argc,               // the argc from command line
    char **argv,            // the argv from command line
    vec2_t *plane_size,     // the size of the plane
    vec2_t *mpi_tasks_grid, // two-uint array defining the MPI tasks' grid
    int *periodic,          // periodic-boundary tag
    int *output_energy_stat,
    int *neighbours,     // four-int array that gives back the
                         // neighbours rank of the calling task
    int *num_iterations, // how many iterations
    int *num_sources,    // how many heat sources
    int *num_local_sources, vec2_t **local_sources,
    double *energy_per_source, // how much heat per source
    plane_t *planes) {
  int halt = 0;
  int ret;
  int verbose = 0;
  int testing = 0;

  // ··································································
  // Set default simulation parameters

  (*plane_size)[_x_] = DEFAULT_GRID_SIZE;
  (*plane_size)[_y_] = DEFAULT_GRID_SIZE;
  *periodic = 0;
  *num_sources = DEFAULT_NUM_SOURCES;
  *num_local_sources = 0;
  *local_sources = NULL;
  *num_iterations = DEFAULT_NUM_ITERATIONS;
  *energy_per_source = DEFAULT_ENERGY_PER_SOURCE;
  *output_energy_stat = 0;

  if (planes == NULL) {
    // Just on prints the error
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - planes pointer is NULL\n", Me);
    return ERROR_NULL_POINTER;
  }

  // Initialize plane dimensions to zero
  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;

  // Set the neighbours to MPI null as default
  for (int i = 0; i < 4; i++)
    neighbours[i] = MPI_PROC_NULL;

  // ··································································
  // process the command line
  while (1) {
    int opt;
    while ((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:t:")) != -1) {
      switch (opt) {
      case 'x':
        (*plane_size)[_x_] = (uint)atoi(optarg);
        break;

      case 'y':
        (*plane_size)[_y_] = (uint)atoi(optarg);
        break;

      case 'e':
        *num_sources = atoi(optarg);
        break;

      case 'E':
        *energy_per_source = atof(optarg);
        break;

      case 'n':
        *num_iterations = atoi(optarg);
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
                 "-x    x size of the plate [%d]\n"
                 "-y    y size of the plate [%d]\n"
                 "-e    number of energy sources [%d]\n"
                 "-E    energy per source [%.1f]\n"
                 "-n    number of iterations [%d]\n"
                 "-o    output energy stats [0 = disabled, 1 = enabled]\n"
                 "-t    testing mode [0 = disabled, 1 = enabled]\n"
                 "-p    periodic boundaries [0 = false, 1 = true]\n"
                 "-v    verbosity level\n"
                 "-h    show this help message\n\n",
                 DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_NUM_SOURCES,
                 DEFAULT_ENERGY_PER_SOURCE, DEFAULT_NUM_ITERATIONS);
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
    return HELP_DISPLAYED;

  // ··································································
  // Comprehensive input validation
  if ((*plane_size)[_x_] <= 0 || (*plane_size)[_y_] <= 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Grid dimensions must be positive, got %d x %d\n",
              (*plane_size)[_x_], (*plane_size)[_y_]);
    return ERROR_INVALID_GRID_SIZE;
  }
  if (*num_sources < 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Number of sources must be >= 0, got %d\n",
              *num_sources);
    return ERROR_INVALID_NUM_SOURCES;
  }
  if (*num_iterations <= 0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Number of iterations must be > 0, got %d\n",
              *num_iterations);
    return ERROR_INVALID_NUM_ITERATIONS;
  }
  if (*energy_per_source <= 0.0) {
    if (Me == 0)
      fprintf(stderr, "ERROR: Energy per source must be > 0.0, got %f\n",
              *energy_per_source);
    return ERROR_INVALID_ENERGY_VALUE;
  }

  // ··································································
  //
  // find a suitable domain decomposition
  // very simple algorithm, you may want to
  // substitute it with a better one
  //
  // the plane Sx x Sy will be solved with a grid
  // of Nx x Ny MPI tasks

  vec2_t grid;
  double formfactor = ((*plane_size)[_x_] >= (*plane_size)[_y_]
                           ? (double)(*plane_size)[_x_] / (*plane_size)[_y_]
                           : (double)(*plane_size)[_y_] / (*plane_size)[_x_]);
  int dimensions = 2 - (num_tasks <= ((int)formfactor + 1));

  if (dimensions == 1) {
    if ((*plane_size)[_x_] >= (*plane_size)[_y_])
      grid[_x_] = num_tasks, grid[_y_] = 1;
    else
      grid[_x_] = 1, grid[_y_] = num_tasks;
  } else {
    int nf;
    uint *factors;
    uint first = 1;
    ret = simple_factorization(num_tasks, &nf, &factors);

    // Find optimal factorization for 2D grid decomposition
    for (int i = 0;
         i < nf && ((double)num_tasks / (first * first) > formfactor); i++)
      first *= factors[i];

    if ((*plane_size)[_x_] > (*plane_size)[_y_])
      grid[_x_] = num_tasks / first, grid[_y_] = first;
    else
      grid[_x_] = first, grid[_y_] = num_tasks / first;
  }

  (*mpi_tasks_grid)[_x_] = grid[_x_];
  (*mpi_tasks_grid)[_y_] = grid[_y_];

  // ··································································
  // my coordinates in the grid of processors based on my rank
  uint X = Me % grid[_x_];
  uint Y = Me / grid[_x_];

  // ··································································
  // find my neighbours
  if (grid[_x_] > 1) {
    if (*periodic) {
      neighbours[EAST] = Y * grid[_x_] + (Me + 1) % grid[_x_];
      neighbours[WEST] =
          (X % grid[_x_] > 0 ? (int)(Me - 1) : (int)((Y + 1) * grid[_x_] - 1));
    }

    else {
      neighbours[EAST] = (X < grid[_x_] - 1 ? Me + 1 : MPI_PROC_NULL);
      neighbours[WEST] = (X > 0 ? (Me - 1) % num_tasks : MPI_PROC_NULL);
    }
  }

  if (grid[_y_] > 1) {
    if (*periodic) {
      neighbours[NORTH] = (num_tasks + Me - grid[_x_]) % num_tasks;
      neighbours[SOUTH] = (num_tasks + Me + grid[_x_]) % num_tasks;
    }

    else {
      neighbours[NORTH] = (Y > 0 ? (int)(Me - grid[_x_]) : MPI_PROC_NULL);
      neighbours[SOUTH] =
          (Y < grid[_y_] - 1 ? (int)(Me + grid[_x_]) : MPI_PROC_NULL);
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
  uint s = (*plane_size)[_x_] / grid[_x_];
  uint r = (*plane_size)[_x_] % grid[_x_];

  // Give extra cells to the first r tasks
  my_size[_x_] = s + (X < r);

  // Do the same for the y axies
  s = (*plane_size)[_y_] / grid[_y_];
  r = (*plane_size)[_y_] % grid[_y_];
  my_size[_y_] = s + (Y < r);

  planes[OLD].size[0] = my_size[0];
  planes[OLD].size[1] = my_size[1];
  planes[NEW].size[0] = my_size[0];
  planes[NEW].size[1] = my_size[1];

  if (verbose > 0) {
    if (Me == 0) {
      printf("Tasks are decomposed in a grid %d x %d\n\n", grid[_x_],
             grid[_y_]);
      fflush(stdout);
    }

    MPI_Barrier(*Comm);

    for (int t = 0; t < num_tasks; t++) {
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
  ret = memory_allocate(planes);

  if (ret != 0) {
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - Memory allocation failed\n", Me);
    return ret;
  }

  // ··································································
  // heat sources are local to the specific patch (thus to the specific rank)
  ret = initialize_sources(Me, num_tasks, Comm, my_size, *num_sources,
                           num_local_sources, local_sources, testing);
  if (ret != 0) {
    if (Me == 0)
      fprintf(stderr, "Rank %d: ERROR - Failed to initialize sources\n", Me);
    return ERROR_INITIALIZE_SOURCES;
  }

  return SUCCESS;
}

// NOTE: This factorization could be leaved to MPI Cartesian Topology but that
// is not really the first optimization to be made considering scalability
// problem with openmp
//
// Think of using a Cartesian decomposition Topology-aware decomposition: With
// the correct topology:
// https://edoras.sdsu.edu/~mthomas/sp17.605/lectures/MPI-Cart-Comms-and-Topos.pdf
uint simple_factorization(uint a, int *n_factorss, uint **factors)
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

  *n_factorss = n;
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

int initialize_sources(int Me, int num_tasks, MPI_Comm *Comm, vec2_t my_size,
                       int num_sources, int *num_local_sources,
                       vec2_t **sources, int testing) {
  // Initialize random number generator with unique seed per process
  srand48(time(NULL) ^ Me);

  int *tasks_with_sources = (int *)malloc(num_sources * sizeof(int));

  if (Me == 0) {
    for (int i = 0; i < num_sources; i++)
      tasks_with_sources[i] = (int)lrand48() % num_tasks;
  }

  MPI_Bcast(tasks_with_sources, num_sources, MPI_INT, 0, *Comm);

  int n_local = 0;
  for (int i = 0; i < num_sources; i++)
    n_local += (tasks_with_sources[i] == Me);
  *num_local_sources = n_local;

  if (n_local > 0) {
    vec2_t *restrict helper = (vec2_t *)malloc(n_local * sizeof(vec2_t));
    for (int s = 0; s < n_local; s++) {
      helper[s][_x_] = 1 + lrand48() % my_size[_x_];
      helper[s][_y_] = 1 + lrand48() % my_size[_y_];
    }

    *sources = helper;

    if (testing != 0) {

      char fname[256];
      snprintf(fname, sizeof(fname), "data_logging/sources_rank%d.txt", Me);
      FILE *f = fopen(fname, "w");
      if (f) {
        for (int s = 0; s < n_local; s++) {
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
int memory_allocate(plane_t *planes_ptr) {
  if (planes_ptr == NULL)
    return ERROR_NULL_POINTER;

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size =
      (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2);

  // Use aligned memory allocation for potential SIMD optimization
  // 64-byte alignment should help with vectorization and cache performance
  if (posix_memalign((void **)&planes_ptr[OLD].data, MEMORY_ALIGNMENT,
                     frame_size * sizeof(double)) != 0) {
    return ERROR_MEMORY_ALLOCATION;
  }

  if (posix_memalign((void **)&planes_ptr[NEW].data, MEMORY_ALIGNMENT,
                     frame_size * sizeof(double)) != 0) {
    free(planes_ptr[OLD].data);
    return ERROR_MEMORY_ALLOCATION;
  }

  // Initialize memory efficiently - memset is faster than nested loops
  memset(planes_ptr[OLD].data, 0, frame_size * sizeof(double));
  memset(planes_ptr[NEW].data, 0, frame_size * sizeof(double));

  // ··················································
  // NOTE: In this case I will use MPI_Datatype directly for strided data
  // NOTE: MPI datatypes are used for direct halo exchange without intermediate buffers
  // North-south communication uses contiguous data (rows)
  // East-west communication uses MPI vector types for strided access (columns)
  // This approach eliminates the need for separate communication buffers

  return SUCCESS;
}

// Release memory also for the buffers
error_code_t memory_release(plane_t *planes_ptr, MPI_Datatype *north_south_type,
                            MPI_Datatype *east_west_type) {
  if (planes_ptr != NULL) {
    if (planes_ptr[OLD].data != NULL)
      free(planes_ptr[OLD].data);

    if (planes_ptr[NEW].data != NULL)
      free(planes_ptr[NEW].data);
  }

  MPI_Type_free(north_south_type);
  MPI_Type_free(east_west_type);

  return SUCCESS;
}

error_code_t output_energy_stat(int step, plane_t *plane_ptr, double budget,
                                int Me, MPI_Comm *Comm) {
  // Set initial energy
  double system_energy = 0;
  double tot_system_energy = 0;
  // Every rank compute its total energy for the patch using parallel reduction
  get_total_energy(plane_ptr, &system_energy);

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
           tot_system_energy / (plane_ptr->size[_x_] * plane_ptr->size[_y_]));
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
