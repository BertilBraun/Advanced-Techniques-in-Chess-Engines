import logging
import os
import time
from mpi4py import MPI

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, filename='multinode_test.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Getting the rank of the process and the size (total number of processes)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Log the start of the script
logging.info(f'Starting multinode test script. Rank: {rank}, Size: {size}')

# Example action: create a file to indicate presence of a node/process
filename = f'node_{rank}_file.txt'
with open(filename, 'w') as f:
    f.write(f'This is a test file created by node {rank}\n')

logging.info(f'File {filename} created by node {rank}')

# Sleep to ensure file visibility across the nodes
time.sleep(10)  # Sleep for 10 seconds

# Checking for file visibility
for i in range(size):
    target_filename = f'node_{i}_file.txt'
    if os.path.exists(target_filename):
        logging.info(f'Node {rank} can see {target_filename}')
    else:
        logging.info(f'Node {rank} cannot see {target_filename}')

# Log the end of the script
logging.info(f'Ending multinode test script. Rank: {rank}')
