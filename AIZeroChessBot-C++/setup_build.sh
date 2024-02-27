#!/bin/bash

#SBATCH --job-name=setup                   # job name
#SBATCH --partition=dev_gpu_4              # mby GPU queue for the resource allocation.
#SBATCH --time=00:05:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --gres=gpu:1
#SBATCH --output=setup_%j.txt
#SBATCH --error=setup_%j.txt

module purge
module load devel/cuda/11.6
module load devel/cmake/3.23.3
module load compiler/gnu/11.2
module load mpi/openmpi/4.1
module save chess

# if conda doesnt have the chess environment, create it
if [ ! -d ~/miniconda3/envs/chess ]; then
    echo "Creating conda environment..."
    conda env create -f environment.yaml
fi

source ~/miniconda3/bin/activate chess


# Set LibTorch download URL
LIBTORCH_URL_CUDA="https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu116.zip"
LIBTORCH_URL_CPU="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip"
LIBTORCH_ZIP="libtorch.zip"

# Create a directory for LibTorch if it doesn't exist
mkdir -p libtorch_cuda
mkdir -p libtorch_cpu

# Download LibTorch only if it hasn't been downloaded yet
if [ ! -f "libtorch_cuda/$LIBTORCH_ZIP" ]; then
    echo "Downloading LibTorch..."
    wget -O "libtorch_cuda/$LIBTORCH_ZIP" "$LIBTORCH_URL_CUDA"
    echo "Extracting LibTorch..."
    unzip -o "libtorch_cuda/$LIBTORCH_ZIP" -d .
fi
if [ ! -f "libtorch_cpu/$LIBTORCH_ZIP" ]; then
    echo "Downloading LibTorch..."
    wget -O "libtorch_cpu/$LIBTORCH_ZIP" "$LIBTORCH_URL_CPU"
    echo "Extracting LibTorch..."
    unzip -o "libtorch_cpu/$LIBTORCH_ZIP" -d .
fi

# Download src/json.hpp if it doesn't exist from https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
if [ ! -f "src/json.hpp" ]; then
    echo "Downloading json.hpp..."
    wget -O "src/json.hpp" "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
fi

# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..
cd ..

# Copy compile_commands.json for IntelliSense (optional)
cp build/compile_commands.json .

echo "Setup completed."

cd train
source train.sh
