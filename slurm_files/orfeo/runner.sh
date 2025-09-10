#!/bin/bash

# List of node counts you want to test
# NODE_LIST=(1 2 4 8)
NODE_LIST=(1 2)

for NODES in "${NODE_LIST[@]}"; do
    echo "Submitting job with $NODES nodes..."
    sbatch --nodes=$NODES mpi_strong_scaling
done
