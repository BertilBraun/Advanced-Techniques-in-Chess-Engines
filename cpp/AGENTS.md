# Local native validation

On Windows, use the WSL Ubuntu toolchain from the repository root. Keep the
build directory in the WSL home directory so dependency downloads and object
files survive between commands:

```bash
cmake -S cpp -B ~/advanced-chess-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_NATIVE_ARCHITECTURE=OFF
cmake --build ~/advanced-chess-build --target AlphaZeroCpp -j2
cmake --build ~/advanced-chess-build --target DirectSelfPlaySearchTest -j2
ctest --test-dir ~/advanced-chess-build -R DirectSelfPlaySearchTest --output-on-failure
```

The extension post-build copy step assumes the build directory is under `cpp/`,
so an external WSL build may finish compilation and linking before that copy
step reports an error.
