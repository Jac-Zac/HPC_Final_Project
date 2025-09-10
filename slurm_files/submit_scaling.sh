#!/bin/bash

# --- Usage / Help ---
usage() {
    echo "Usage: $0 <cineca|orfeo> <strong|weak>"
    echo ""
    echo "  cineca|orfeo    : Target system directory where the Slurm scripts are located"
    echo "  strong|weak     : Type of scaling test to run"
    echo ""
    echo "Example:"
    echo "  $0 orfeo strong"
    exit 1
}

# --- Validate arguments ---
if [ $# -ne 2 ]; then
    usage
fi

SYSTEM=$1
SCALING=$2

if [[ "$SYSTEM" != "cineca" && "$SYSTEM" != "orfeo" ]]; then
    echo "Error: SYSTEM must be 'cineca' or 'orfeo'"
    usage
fi

if [[ "$SCALING" != "strong" && "$SCALING" != "weak" ]]; then
    echo "Error: SCALING must be 'strong' or 'weak'"
    usage
fi

# --- Node counts to test ---
NODE_LIST=(1 2)

# --- Submit jobs ---
for NODES in "${NODE_LIST[@]}"; do
    JOB_FILE="slurm_files/$SYSTEM/mpi_${SCALING}_scaling"
    if [ ! -f "$JOB_FILE" ]; then
        echo "Error: Job file '$JOB_FILE' does not exist!"
        exit 1
    fi

    echo "Submitting $SCALING scaling job with $NODES nodes from $SYSTEM..."
    sbatch --nodes=$NODES "$JOB_FILE"
done
