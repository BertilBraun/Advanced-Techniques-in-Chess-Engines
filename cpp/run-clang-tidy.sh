#!/bin/bash

SRC_DIR=src
BUILD_DIR=build

# Process .hpp files first
find $SRC_DIR -name "*.hpp" | while read file; do
    # Exclude chess.hpp and json.hpp
    if [[ "$file" == *"src/chess.hpp"* || "$file" == *"src/util/json.hpp"* ]]; then
        continue
    fi
    echo "Running clang-tidy on $file"
    clang-tidy "$file" -p "$BUILD_DIR" --fix --format-style=file
done

# Then process .cpp files
find $SRC_DIR -name "*.cpp" | while read file; do
    # Exclude binding.cpp
    if [[ "$file" == *"src/binding.cpp"* ]]; then
        continue
    fi
    echo "Running clang-tidy on $file"
    clang-tidy "$file" -p "$BUILD_DIR" --fix --format-style=file
done
