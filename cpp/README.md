# C++ Self-Play Engine for AlphaZero-Like Chess Bot

This project is a high-performance C++ backend for an AlphaZero-style chess engine, focused specifically on the computationally intensive MCTS components.  
It supplements the Python-based system by providing a faster, parallelized multithreaded MCTS engine that can be directly called from Python via PyBind bindings.

While originally intended to replace the Python version entirely, some higher-level logic remains simpler and more flexible in Python. Therefore, the C++ component now permanently acts as a powerful backend for the Python project.

![AlphaZero Chess Engine Architecture](../documentation/images/C++%20Overview.png)

## Documentation

Detailed documentation for each project component can be found in:

- **[Chess Encoding for Neural Networks](https://deepwiki.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/5.1-chess-implementation)**: How chess board states are encoded as inputs to the neural network.
- **[Chess Framework](https://github.com/BertilBraun/Stockfish)**: Details about the the chess framework and logic adapted from the Stockfish chess engine.
- **[MCTS Parallelization](https://deepwiki.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/4.1-c++-mcts-engine)**: Strategies used for efficient parallel MCTS.
- **[Inference Client](https://deepwiki.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/4.2-c++-inference-client)**: Architecture of the C++ inference client.

## Technologies

- **Programming Language**: C++20
- **Machine Learning Frameworks**: LibTorch (PyTorch C++ API)
- **Chess Library**: Stockfish adapted chess engine for board logic [see here](https://github.com/BertilBraun/Stockfish)
- **Python Integration**: PyBind11 for binding C++ modules into Python
- **Model Sharing**: Neural network models are defined in Python, JIT-compiled with PyTorch, and loaded into C++ at runtime

## Architecture Overview

- **MCTS Engine**: Written in C++, highly parallelized.
- **Bindings**: C++ exports functionality (e.g., starting MCTS searches, inspecting search tree nodes, etc.) to Python via PyBind11.
- **Data Interface**: Data exchange between Python and C++ happens through pybind bindings of the C++ STL data structures, allowing efficient transfer of game states and MCTS results.
- **Model Loading**: Python-side model definitions are JIT-exported to `.jit.pt` files, which the C++ engine loads directly with LibTorch.

All orchestration (job submission, training loop, etc.) happens in Python — the C++ part provides the raw MCTS data.

## Getting Started

### Step 1: Build the Project

Ensure you have PyTorch installed via pip as well as a compatible C++ compiler (e.g., GCC, Clang, MSVC) and CMake.
You can use the provided `py/setup.sh` scripts to automate downloading dependencies and generating build files.

```bash
compile
or
compileDebug
```

which expands to:

```bash
cd ../cpp && mkdir -p build && cd build && \
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3.10) && \
make -j && cd ../../py
```

The build process generates a shared object file:

- `py/AlphaZeroCpp.so` (Linux/Mac)
- `py/AlphaZeroCpp.pyd` (Windows)

As well as type stubs for Python:

- `py/AlphaZeroCpp.pyi`

This shared object can be imported as a Python module.

### Step 2: Using from Python

Once built, the module can be imported in your Python code:

```python
import AlphaZeroCpp

mcts = AlphaZeroCpp.MCTS(...) # initialize with parameters
# search with fens
res = mcts.search(["rkbq...BKR w KQkq - 0 1", "rkbq...BKR b KQkq - 0 1"]) # search multiple positions in parallel
print(res)  # prints the results of the MCTS search
```

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

## Performance Considerations

Preliminary testing notes, that the C++ implementation of the pure MCTS search is 90-100x faster than the Python implemenation
    Python: 100 iterations took 232.29s
    C++: 100 Iterations took 2.359s

## Summary

- The C++ engine handles **parallel MCTS**.
- **Python** handles **training**, **evaluation**, and **cluster orchestration**.
- Communication happens via **PyBind bindings**.
- Models are shared via **PyTorch JIT** export/import.
