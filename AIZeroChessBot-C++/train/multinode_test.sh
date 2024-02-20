#!/bin/bash

#SBATCH --job-name=multinode_test          # job name
#SBATCH --partition=multiple               # mby GPU queue for the resource allocation.
#SBATCH --time=00:05:00                    # wall-clock time limit
#SBATCH --mem=2000                         # memory per node
#SBATCH --nodes=2                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=3                # maximum count of tasks per node
#SBATCH --output=multinode_test.txt
#SBATCH --error=multinode_test.txt

srun python multinode_test.py
