#!/bin/bash
#BATCH --nodes=1                   # 1 node
#SBATCH --cpus-per-task=1          # serial run, 1 CPU is enough
#SBATCH --ntasks-per-node=1        # 1 task
#SBATCH --mem=0                    # use all memory
#SBATCH --partition=THIN
##SBATCH -A uTS25_Tornator_0       # your project/account
#SBATCH -t 00:00:22                # 22 seconds
#SBATCH --job-name=stencil_serial

EXEC=./stencil_serial              # your executable

# =======================================================
export OMP_PLACES=threads
export OMP_PROC_BIND=close

# Run the code
${EXEC} -x 5000 -y 5000 -n 500 > out_stencil_serial.log
