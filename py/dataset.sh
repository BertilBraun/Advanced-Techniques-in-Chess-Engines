#!/bin/bash

#SBATCH --job-name=dataset              # job name
#SBATCH --partition=cpuonly            # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=8                # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --output=dataset.txt
#SBATCH --error=dataset.txt

source setup.sh

python -m src.games.chess.ChessDatabase 2024 2 10