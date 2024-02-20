#!/bin/bash

#SBATCH --job-name=train_zero              # job name
#SBATCH --partition=gpu_8                  # mby GPU queue for the resource allocation.
#SBATCH --time=06:10:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=6                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_zero_%j.txt
#SBATCH --error=train_zero_%j.txt

module load devel/cuda/11.6
module load devel/cmake/3.23.3
module load mpi/openmpi/4.1


source ~/miniconda3/bin/activate chess

# Start training via ./AIZeroChessBot, timeout after 12 hours, then requeue the job
timeout 12h ./AIZeroChessBot

sbatch train.sh

