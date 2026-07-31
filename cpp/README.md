# Native multi-game engine foundation

The C++20 rework now separates common contracts (`az_core`), generic search
infrastructure (`az_search`), and the Go specialization (`az_go`). Go rules,
state mutation, encoding, and the predecessor fixed PUCT implementation are
native. Go supports arbitrary square board sizes from 3 upward, subject only
to the signed 32-bit action-space representation.

The common game/session contracts and bounded tree arena are foundations for
both Go and chess. Native chess rules, search execution, LibTorch inference,
and complete multi-game sessions remain pending.

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

The current Python module remains `az_go_native`. Its
`PythonGoEvaluator` callback submits typed leaf requests to Python's per-device
inference broker. This is the predecessor Go path, not the target production
architecture; the target keeps complete self-play search, batching, and
LibTorch inference in C++.

These commands perform structural compilation and unit tests only. Do not run
training, performance benchmarks, or prolonged GPU work until the target
compute environment is explicitly available.

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
compiler-option changes so `az_core`, `az_go`, `az_search`, their tests, and
enabled bindings are visible to IDE and clang-tidy tooling. Format with the
repository `.clang-format` and run clang-tidy with `.clang-tidy`.

## VS Code IntelliSense

The repository workspace configures the Microsoft C/C++ extension for both
native Windows and WSL:

- `Windows-Clang` uses the MSYS2 Clang installation and forces the project PCH,
  so aliases from `common.hpp` are available in translation units that receive
  it from CMake.
- `WSL-Clang` uses `build-go-clang/compile_commands.json`. Open the repository
  through the VS Code WSL extension before selecting this configuration.

Do not use a WSL-generated `compile_commands.json` from native Windows VS Code.
Its `/mnt/c/...` paths and Linux compiler are not valid Windows IntelliSense
inputs. Run **C/C++: Select IntelliSense Configuration** if VS Code does not
select the configuration for the current host automatically, then run
**Developer: Reload Window**.
