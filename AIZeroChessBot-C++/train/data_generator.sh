#!/bin/bash

#SBATCH --job-name=generator               # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=02:10:00                    # wall-clock time limit
#SBATCH --mem=80000                        # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=20                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=generator_%j.txt
#SBATCH --error=generator_%j.txt

module restore chess

source ~/miniconda3/bin/activate chess

cd ../build

cmake --build . --config Release

cp AIZeroChessBot ../train/AIZeroChessBot

cd ../train


./AIZeroChessBot "generate" $SLURM_CPUS_PER_TASK
exit_status=$?

# Check if the process was successful (exit status 0)
if [ $exit_status -eq 0 ]; then
    echo "Process completed successfully, re-queuing..."
    sbatch train.sh
else
    echo "Process did not complete successfully, not re-queuing."
fi
