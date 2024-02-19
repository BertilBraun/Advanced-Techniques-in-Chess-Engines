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

module load devel/cuda/11.6
module load devel/cmake/3.23.3

export PATH=~/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=~/miniconda3/envs/chess/bin:$PATH
export LD_LIBRARY_PATH=~/miniconda3/envs/chess/lib:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate chess

cd ../build

cmake --build . --config Release

cd Release

./AIZeroChessBot
