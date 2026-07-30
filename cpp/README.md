# Native Go rules and search

The C++20 library is the only production implementation of Go state mutation
and MCTS. It supports arbitrary square board sizes from 3 upward, subject only
to the signed 32-bit action-space representation.

`src/common.hpp` is the project precompiled header. Native code uses its
`uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, and `int64`
aliases.

## Build tests and Python bindings

From the repository root:

```powershell
cmake -S .\cpp -B .\cpp\build -DAZ_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .\cpp\build --parallel
ctest --test-dir .\cpp\build --output-on-failure
```

The Python module is `az_go_native`. Native search owns traversal, selection,
expansion, and backup. Its `PythonGoEvaluator` callback submits typed leaf
requests to Python's per-device inference broker, which batches PyTorch model
execution. There is no LibTorch model owner or batching thread in C++.

## Clang tooling and IDE database

Use a separate Clang build so clang-tidy reads a compatible PCH:

```powershell
cmake -S .\cpp -B .\cpp\build-clang `
    -DAZ_BUILD_PYTHON=ON `
    -DCMAKE_BUILD_TYPE=RelWithDebInfo `
    -DCMAKE_CXX_COMPILER=clang++
cmake --build .\cpp\build-clang --parallel
Copy-Item .\cpp\build-clang\compile_commands.json .\cpp\compile_commands.json
```

`compile_commands.json` is ignored but should be regenerated after target or
compiler-option changes. Format with the repository `.clang-format` and run
clang-tidy with `.clang-tidy`.
