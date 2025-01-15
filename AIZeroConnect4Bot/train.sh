#!/bin/bash

#SBATCH --job-name=train_zero              # job name
#SBATCH --partition=accelerated            # mby GPU queue for the resource allocation.
#SBATCH --time=04:00:00                    # wall-clock time limit
#SBATCH --mem=512000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=152                # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:4
#SBATCH --output=train_zero_%j.txt
#SBATCH --error=train_zero_%j.txt

source setup.sh

python3.11 train.py