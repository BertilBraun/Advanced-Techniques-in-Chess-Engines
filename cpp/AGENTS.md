# Local native compilation

On Windows, use the WSL Ubuntu toolchain from the repository root. Install
`ccache` once inside WSL:

```bash
sudo apt-get update
sudo apt-get install -y ccache
```

For a routine compile check, configure the persistent WSL build directory once
with the unoptimized `CompileCheck` build type, then build only the Python
extension. Do not enable tests or benchmarks for this check:

```bash
cmake -S cpp -B ~/advanced-chess-compile-check \
  -DCMAKE_BUILD_TYPE=CompileCheck \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DENABLE_NATIVE_ARCHITECTURE=OFF
cmake --build ~/advanced-chess-compile-check --target AlphaZeroCpp --parallel "$(nproc)"
```

Reuse that build directory so `ccache` and unchanged object files can serve
later checks. Reconfigure only when CMake configuration or dependencies change.
Confirm cache use with `ccache --show-stats`. CompileCheck links the extension
inside the build directory but does not copy it into `py` or regenerate stubs.

Native tests and benchmarks are opt-in. Use a separate build directory so
enabling them does not expand the routine compile-check graph:

```bash
cmake -S cpp -B ~/advanced-chess-tests \
  -DCMAKE_BUILD_TYPE=CompileCheck \
  -DBUILD_TESTING=ON \
  -DBUILD_BENCHMARKS=ON \
  -DENABLE_NATIVE_ARCHITECTURE=OFF
cmake --build ~/advanced-chess-tests --parallel "$(nproc)"
ctest --test-dir ~/advanced-chess-tests --output-on-failure
```

Production artifacts still require an explicit Release build; a CompileCheck
artifact must never be deployed or used for performance measurements.
