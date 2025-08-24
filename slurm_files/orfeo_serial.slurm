#!/bin/bash
#SBATCH --nodes=1                   # 1 node
#SBATCH --cpus-per-task=8           # 8 OpenMP threads per MPI task
#SBATCH --ntasks-per-node=1         # 4 MPI tasks on the node
#SBATCH --mem=0                     # use all available memory
#SBATCH --partition=THIN
#SBATCH -t 00:00:30                 # 5 minutes
#SBATCH --job-name=stencil_test

# Set the executable name
EXEC=./stencil_serial

# =======================================================
# Set OpenMP environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Set the total number of MPI tasks
TOTAL_MPI_TASKS=${SLURM_NTASKS}

# Define the arguments for the executable
# A medium-sized grid suitable for a single-node run
ARGS="-x 5000 -y 5000 -n 50"

# Print run information
echo "Running on a single node with ${TOTAL_MPI_TASKS} MPI tasks."
echo "Each task uses ${OMP_NUM_THREADS} OpenMP threads."

# Run the code using mpirun
# The output will be saved to a file named 'out_orfeo_parallel.log'
mpirun -np ${TOTAL_MPI_TASKS} ${EXEC} ${ARGS} > out_orfeo_parallel.log

