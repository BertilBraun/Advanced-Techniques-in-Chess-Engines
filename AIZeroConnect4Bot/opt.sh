#!/bin/bash

#SBATCH --job-name=opt_zero                # job name
#SBATCH --partition=dev_accelerated        # mby GPU queue for the resource allocation.
#SBATCH --time=01:00:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1                       # number of GPUs required
#SBATCH --output=opt_zero_%j.txt
#SBATCH --error=opt_zero_%j.txt

source setup.sh

python3.11 -m src.hyperparameter_optimization.opt &
python3.11 -m src.hyperparameter_optimization.opt &
python3.11 -m src.hyperparameter_optimization.opt &
python3.11 -m src.hyperparameter_optimization.opt