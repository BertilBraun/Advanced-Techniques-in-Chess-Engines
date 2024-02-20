#!/bin/bash

#SBATCH --job-name=multinode_test          # job name
#SBATCH --partition=multiple               # mby GPU queue for the resource allocation.
#SBATCH --time=00:05:00                    # wall-clock time limit
#SBATCH --mem=2000                         # memory per node
#SBATCH --nodes=2                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=3                # maximum count of tasks per node
#SBATCH --output=multinode_test.txt
#SBATCH --error=multinode_test.txt

module load compiler/gnu/11.2
module load mpi/openmpi/4.1

mpirun -np 6 python multinode_test.py
