#!/bin/bash

SRC_DIR=src
BUILD_DIR=build

find $SRC_DIR \( -name "*.cpp" -o -name "*.hpp" \) | while read file; do
    echo "Running clang-tidy on $file"
    clang-tidy "$file" -p "$BUILD_DIR" --fix --format-style=file
done
