#!/bin/bash

#SBATCH --job-name=train_zero              # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=06:00:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=2                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_zero_%j.txt
#SBATCH --error=train_zero_%j.txt

module load devel/cuda/11.8
module load devel/miniconda/4.9.2
module load mpi/openmpi/default
module load devel/cmake/3.23.3

conda activate chess


cd ../build

cmake --build . --config Release

cd Release

./AIZeroChessBot
