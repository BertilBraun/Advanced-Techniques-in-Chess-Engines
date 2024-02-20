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
module load compiler/gnu/11.2
module load mpi/openmpi/4.1

source ~/miniconda3/bin/activate chess

cd ../build

cmake --build . --config Release

cp AIZeroChessBot ../train/AIZeroChessBot

cd ../train

# Explanation:
# We are running the communicator.py file with 6 processes.
# The communicator.py file is responsible for the communication between the training and the self-play processes.
# The root process is responsible for the training process, while the other 5 processes are responsible for the self-play process.
# Refer to the README.md file for information about the relationship between the number of workers and the training process.
#
# The timeout command is used to kill the communicator.py process after 6 hours.
# This is done to be able to requeue the job on the cluster to continue the training process.

timeout 6h mpirun -np 6 python communicator.py
exit_status=$?

# Check if the process was successful (exit status 0)
if [ $exit_status -eq 0 ]; then
    echo "Process completed successfully, re-queuing..."
    sbatch train.sh
else
    echo "Process did not complete successfully, not re-queuing."
fi
