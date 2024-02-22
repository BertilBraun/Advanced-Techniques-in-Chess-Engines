#!/bin/bash

#SBATCH --job-name=generator               # job name
#SBATCH --partition=single                 # mby GPU queue for the resource allocation.
#SBATCH --time=12:10:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --output=generator_%j.txt
#SBATCH --error=generator_%j.txt

module restore chess

source ~/miniconda3/bin/activate chess

cd ../build

cmake --build . --config Release

cp AIZeroChessBot ../train/AIZeroChessBot

cd ../train

./AIZeroChessBot generator 1 1
exit_status=$?

# Check if the process was successful (exit status 0)
if [ $exit_status -eq 0 ]; then
    echo "Process completed successfully, re-queuing..."
    sbatch data_generator.sh
else
    echo "Process did not complete successfully, not re-queuing."
fi