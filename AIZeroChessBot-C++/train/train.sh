#!/bin/bash

#SBATCH --job-name=train_zero              # job name
#SBATCH --partition=gpu_8                  # mby GPU queue for the resource allocation.
#SBATCH --time=06:10:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=32                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_zero_%j.txt
#SBATCH --error=train_zero_%j.txt

module restore chess

source ~/miniconda3/bin/activate chess

cd ../build

cmake --build . --config Release

cp AIZeroChessBot ../train/AIZeroChessBot

cd ../train

# Extract the SLURM job time limit in minutes and convert to seconds for timeout
# Subtract a buffer time (e.g., 300 seconds) to allow for cleanup and requeueing
TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
# Convert hh:mm:ss to seconds
job_time_limit_seconds=$(echo $TIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
# Subtract buffer time
job_time_limit_seconds=$(($job_time_limit_seconds - 300))

# The timeout command is used to kill the communicator.py process after job_time_limit_seconds.
# This is done to be able to requeue the job on the cluster to continue the training process.

# Use the calculated job time limit for timeout
timeout ${job_time_limit_seconds}s ./AIZeroChessBot "train" $SLURM_CPUS_PER_TASK
exit_status=$?


# Check if the process was successful (exit status 0)
if [ $exit_status -eq 0 ]; then
    echo "Process completed successfully, re-queuing..."
    sbatch train.sh
else
    echo "Process did not complete successfully, not re-queuing."
fi
