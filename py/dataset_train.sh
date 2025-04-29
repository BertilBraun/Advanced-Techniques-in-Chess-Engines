#!/bin/bash

#SBATCH --job-name=dataset_train              # job name
#SBATCH --partition=accelerated            # mby GPU queue for the resource allocation.
#SBATCH --time=04:00:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=2                # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=dataset_train_%j.txt
#SBATCH --error=dataset_train_%j.txt

source setup.sh

python -m src.games.ChessDatabase 30 2000

python -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5 