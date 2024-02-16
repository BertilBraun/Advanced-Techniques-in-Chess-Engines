#!/bin/bash

# Set LibTorch download URL
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.8.1%2Bcpu.zip"
LIBTORCH_ZIP="libtorch.zip"

# Create a directory for LibTorch if it doesn't exist
mkdir -p libtorch

# Download LibTorch only if it hasn't been downloaded yet
if [ ! -f "libtorch/$LIBTORCH_ZIP" ]; then
    echo "Downloading LibTorch..."
    wget -O "libtorch/$LIBTORCH_ZIP" "$LIBTORCH_URL"
    echo "Extracting LibTorch..."
    unzip -o "libtorch/$LIBTORCH_ZIP" -d libtorch
fi

# Create a build directory
mkdir -p build
cd build

# Run CMake to configure the project. Adjust the path to your LibTorch cmake folder
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DTorch_DIR=$(pwd)/../libtorch/share/cmake/Torch ..

cd ..

# Copy compile_commands.json for IntelliSense (optional)
cp build/compile_commands.json .

echo "Setup completed."
