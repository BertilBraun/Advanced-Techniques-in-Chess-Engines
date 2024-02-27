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
module restore chess

# The Structure after the setup is done is as follows:
# eval
# ├── build
# │   └── EvalAIZeroChessBot
# ├── models
# │   └── stockfish-8-linux
# ├── EvalAIZeroChessBot
# └── eval_setup_build.sh

# Download the model file
mkdir -p models

# Download the stockfish model file only if models/stockfish_8_x64 doesn't exist
if [ ! -f "models/stockfish_8_x64" ]; then
    echo "Downloading stockfish model..."
    wget -O models/stockfish.zip https://drive.usercontent.google.com/u/0/uc?id=1hhuy4O9grrqaL92hbFboMrwFnCZXNheq\&export=download
    unzip models/stockfish.zip -d models
    mv models/stockfish-8-linux/Linux/stockfish_8_x64 models/stockfish_8_x64
    rm -rf models/stockfish-8-linux
    rm models/stockfish.zip
else
    echo "Stockfish model already downloaded."
fi

# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..

cmake --build . --config Release

cp EvalAIZeroChessBot ../EvalAIZeroChessBot

cd ..

echo "Setup completed."
