import os
from mpi4py import MPI

# Getting the rank of the process and the size (total number of processes)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    exit(os.system(f'./AIZeroChessBot "root" {rank} {size} >> trainer.log'))
else:
    exit(os.system(f'./AIZeroChessBot "worker" {rank} {size} >> worker_{rank}.log'))
