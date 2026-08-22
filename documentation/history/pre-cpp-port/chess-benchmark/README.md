# Chess Engine Benchmarking

## Introduction

The goal of this benchmark is to compare the performance of the C++ chess implementation with the python one. The C++ implementation is a direct translation of the python one, so the algorithms and data structures are the same. The only difference is the language and the way the code is executed.

The reason for the translation to C++ is to improve the performance of the chess engine. The python implementation is slow, and the MCTS search was taking 90% of the time in the python implementation while the goal is to have the NN taking 90% of the time and the MCTS 10% of the time.

## Benchmarks

### 1. Move Generation Benchmark

This benchmark measures the time taken to generate all legal moves up to a specified depth. The depth parameter controls the complexity of the move generation task, with higher values leading to an exponential increase in the number of positions explored. The benchmark records both the total time taken to perform the operation across multiple iterations and the average time per iteration.

### 2. Board State Copy Benchmark

This benchmark evaluates the performance of copying the entire game state, a common operation in chess engines that allows for move simulation and undoing. The benchmark records the time taken to copy the board state multiple times, providing both total and average times for the operation.

### Parameters

- **Iterations**: The number of times the benchmark operation is performed. Higher iterations increase the accuracy of the timing measurement but require more time to complete. A default value of `100` iterations is used to balance accuracy and execution time.
- **Depth**: Specifies how many moves ahead the move generation benchmark should explore. A depth of `4` is used as a default for a balance between complexity and execution time.

## Results

### 1. Move Generation Benchmark Results

- **C++ Implementation**
  - **Iterations**: 100
  - **Depth**: 4
  - **Total Time**: 3.391 seconds
- **Python Implementation**
  - **Iterations**: 100
  - **Depth**: 4
  - **Total Time**: 149.688 seconds

### 2. Board State Copy Benchmark Results

- **C++ Implementation**
  - **Iterations**: 100.000.000
  - **Total Time**: 1.207 seconds
- **Python Implementation**
  - **Iterations**: 100.000.000
  - **Total Time**: 131.296 seconds

## Conclusion

The benchmark results show that the C++ implementation is significantly faster than the python one. This means that the C++ implementation is a good candidate for the chess engine, and the next step is to integrate the neural network and the MCTS search to the C++ implementation and compare the performance with the python implementation.
