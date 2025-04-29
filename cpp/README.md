# C++ Self-Play Engine for AlphaZero-Like Chess Bot

This project is a high-performance C++ backend for an AlphaZero-style chess engine, focused specifically on the computationally intensive self-play and training data generation components.  
It supplements the Python-based system by providing a faster, parallelized self-play engine that can be directly called from Python via PyBind bindings.

While originally intended to replace the Python version entirely, some higher-level logic remains simpler and more flexible in Python. Therefore, the C++ component now permanently acts as a powerful backend for the Python project.

## Documentation

Detailed documentation for each project component can be found in:

- **[Chess Encoding for Neural Networks](documentation/implementation/encodings.md)**: How chess board states are encoded as inputs to the neural network.
- **[Chess Framework](documentation/implementation/chess/README.md)**: Details about the ported chess logic and performance gains.
- **[Pre-Training System](documentation/optimizations/pretraining.md)**: Using grandmaster games and Stockfish to bootstrap the initial training.
- **[Parallelization](documentation/implementation/parallelization/README.md)**: Strategies used for efficient parallel self-play and data generation.

## Technologies

- **Programming Language**: C++20
- **Machine Learning Frameworks**: LibTorch (PyTorch C++ API)
- **Chess Library**: Port of `python-chess` to C++
- **Python Integration**: PyBind11 for binding C++ modules into Python
- **Model Sharing**: Neural network models are defined in Python, JIT-compiled with PyTorch, and loaded into C++ at runtime

## Architecture Overview

- **Self-Play Engine**: Written in C++, highly parallelized.
- **Bindings**: C++ exports functionality (e.g., starting self-play workers) to Python via PyBind11.
- **Data Interface**: Self-play workers write training data to files. File formats are consistent between C++ and Python.
- **Model Loading**: Python-side model definitions are JIT-exported to `.jit.pt` files, which the C++ engine loads directly with LibTorch.

All orchestration (job submission, training loop, etc.) happens in Python — the C++ part provides the raw self-play data.

## Getting Started

### Step 1: Build the Project

Ensure you have PyTorch installed via pip as well as a compatible C++ compiler (e.g., GCC, Clang, MSVC) and CMake.
You can use the provided setup scripts to automate downloading dependencies and generating build files.

```bash
cd build
cmake --build . --config Release
```
or
```bash
cd build
make
```

The build process generates a shared object file:

- `cpp_py/AlphaZeroCpp.so` (Linux/Mac)
- `cpp_py/AlphaZeroCpp.pyd` (Windows)

This shared object can be imported as a Python module.

### Step 2: Using from Python

Once built, the module can be imported in your Python code:

```python
import AlphaZeroCpp

# Example: start self-play workers
AlphaZeroCpp.start_self_play(num_workers=8, model_path="path/to/model.jit.pt")
```

The self-play workers will generate training data files, which can then be loaded by the Python-side training logic.

**Note**: The neural network model must be exported from Python using PyTorch’s JIT export (`torch.jit.save`) to ensure compatibility.

### Step 3: Model Export (Python Side)

Before starting self-play, you must save your model in a format the C++ engine can load:

```python
import torch

model = YourModel()
# After loading or training
traced_model = torch.jit.trace(model, example_inputs)
torch.jit.save(traced_model, "path/to/model.jit.pt")
```

## Cluster Usage

The C++ code is no longer directly submitted to the cluster for execution.  
All cluster job management, distributed training orchestration, and evaluation pipelines are handled via the Python system (`cpp_py/`).

Refer to the main Python project’s README for instructions on how to launch cluster jobs.

## Summary

- The C++ engine handles **self-play** and **data generation**.
- **Python** handles **training**, **evaluation**, and **cluster orchestration**.
- Communication happens via **file-based data exchange** and **PyBind bindings**.
- Models are shared via **PyTorch JIT** export/import.
