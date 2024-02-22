import os
import sys
from mpi4py import MPI

is_training = sys.argv[1] == "train"
is_generating = sys.argv[1] == "generate"

if not is_training and not is_generating:
    print("Invalid mode ", sys.argv[1])
    exit(1)

# Getting the rank of the process and the size (total number of processes)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if is_training:
    print("Training mode")

    if rank == 0:
        exit(os.system(f'./AIZeroChessBot "root" {rank} {size} >> trainer.log'))
    else:
        exit(os.system(f'./AIZeroChessBot "worker" {rank} {size} >> worker_{rank}.log'))

if is_generating:
    print("Generating mode")
    exit(os.system(f'./AIZeroChessBot "generator" {rank} {size} >> generator.log'))