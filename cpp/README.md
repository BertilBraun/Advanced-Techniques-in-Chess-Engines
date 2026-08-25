# Native game, inference, and search runtime

The C++20 extension is the authoritative runtime for chess and Go rules, immutable state transitions, packed neural
network encoding, batched TorchScript inference, tree search, self-play search, and retained-root interactive
analysis. Python supervises experiments and consumes this coarse native boundary; there is no Python MCTS or second
native search implementation.

## Source ownership

- `src/games/chess/`: Stockfish-backed chess rules, action mapping, packed encoding, bindings, and UCI-facing
  presentation;
- `src/games/go/`: compile-time 7x7 and 9x9 Go rules, positions, action mapping, encoding, and bindings;
- `src/search/`: shared inference pipeline, arena tree, batched executor, self-play facade, and analysis facade;
- `src/util/`: shared packed planes, fixed-size bitboards, logging, and timing mechanics;
- `test/`: the unified native correctness and binding-contract suite;
- `benchmark/`: standalone native inference benchmark target.

The optimized search is game-parameterized at compile time and instantiated for chess, Go 7x7, and Go 9x9. Python
bindings expose the same action-ID root/search surface for every game.

## Build

Install the hashed Python environment first; it supplies PyTorch/LibTorch and `pybind11-stubgen`. Then configure and
build from the repository root:

```powershell
uv pip install --require-hashes --requirements .\py\requirements-training.lock
cmake -S .\cpp -B .\cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build .\cpp\build --parallel
```

On a fresh compute node, [`deployment/setup_remote.sh`](../deployment/setup_remote.sh) performs the locked install
and Release build before starting the supplied runner command.

The ordinary build copies `AlphaZeroCpp.so` into `py/`, regenerates `py/AlphaZeroCpp.pyi` from the bindings and
restyles it with `ruff check --fix` and `ruff format`, so a rebuild of unchanged bindings leaves the checkout clean
and `deployment/run_control.sh` can start. This needs `ruff` in the build interpreter's environment (the `dev`
dependency group); configuration fails with a clear message if it is missing. Production
inference loads the trimmed TorchScript policy/WDL artifact published by the Python checkpoint writer.

## Validation

Run the native suite after a successful build:

```powershell
ctest --test-dir .\cpp\build --output-on-failure
```

Then run the exact Python suite from `py` so native-backed tests import the freshly built extension:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Changes to bindings, state, encoding, inference, or search require both suites. Historical performance evidence is
under [`documentation/benchmarks`](../documentation/benchmarks/README.md); it is revision-specific and not current
architecture guidance.
