# AlphaZero chess and Go experimentation platform

This repository is a typed, configuration-driven AlphaZero research runtime for chess, 7x7 Go, and 9x9 Go. Native
C++ owns game rules, packed network encoding, batched inference, tree search, and interactive analysis. Python owns
experiment composition, self-play supervision, memory-mapped replay, persistent DDP training, elapsed evaluation,
telemetry, and artifact management.

The original chess project produced an approximately 2000-2100 Elo model on a modest personal research budget.
Those models, games, plots, and logs remain available as
[historical result evidence](documentation/benchmarks/chess-results/); they do not describe the current runtime.

## Current architecture

One resolved experiment configuration selects a concrete chess or Go implementation. Both games then use the same
production lifecycle:

1. persistent workers run the shared native self-play search and publish atomic completed trajectories;
2. the coordinator drains those trajectories into one fixed-slot circular memory-mapped replay;
3. persistent symmetric DDP ranks train one blocking optimizer quantum from read-only mapped replay;
4. rank zero publishes the complete checkpoint and trimmed policy/WDL inference artifact;
5. workers transition to the new generation, while short-lived evaluation jobs run on fixed elapsed boundaries.

There is no Python search implementation, alternate replay/trainer architecture, or generation-gating evaluator.
Chess and Go evaluation share the same manager, paired-match runner, fixed-dataset metrics, result artifacts, and
reporting path. Stockfish and KataGo remain explicitly configured external processes.

For the authoritative design and current phase status, start at the
[documentation index](documentation/README.md). `THINGS_TO_TRY.md` is the experiment backlog, not an authorization
ledger.

## Repository layout

- `cpp/`: native chess/Go state, encoding, inference, search, bindings, benchmarks, and tests;
- `py/`: experiment configuration, coordinator, replay, training, evaluation, UCI, and optional tools;
- `deployment/`: fresh-node bootstrap plus web and Lichess deployment assets;
- `documentation/architecture/`: current authoritative plans and architecture;
- `documentation/operations/`: current deployment and operations guidance;
- `documentation/history/`: explicitly non-normative implementation history;
- `documentation/benchmarks/`: historical benchmark and result evidence.

## Fresh compute-node setup

[`deployment/setup_remote.sh`](deployment/setup_remote.sh) is the authoritative bootstrap for a fresh training node.
It clones the requested revision, installs the hashed training environment, builds the Release extension, exports
`ENGINE_SOURCE_REVISION`, and executes the supplied command:

```bash
curl -fsSL https://raw.githubusercontent.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/master/deployment/setup_remote.sh \
  | bash -s -- bash -c 'python py/train.py \
      --run-config /data/approved-experiment.yaml \
      --expected-source-revision "$ENGINE_SOURCE_REVISION" \
      --approval-file /data/run-approval.json'
```

Set `ENGINE_REPOSITORY_REF`, `ENGINE_REPOSITORY_DIRECTORY`, `ENGINE_VIRTUAL_ENVIRONMENT`, or
`ENGINE_REPOSITORY_URL` to override checkout/environment locations. The script intentionally does not install
Stockfish, KataGo, or their model/configuration artifacts.

The checked-in `py/configs/*-experiment-template.yaml` files are validation templates, not approved production
runs. Hardware, artifact paths and hashes, output paths, source revision, and approval must be resolved explicitly.

## Validation

Run Python validation from `py`:

```powershell
uv run ruff format
uv run ruff check --fix
python -m pytest --import-mode=importlib .\test -q
```

Build and test native code from the repository root:

```powershell
cmake -S .\cpp -B .\cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build .\cpp\build --parallel
ctest --test-dir .\cpp\build --output-on-failure
```

Native-facing Python tests require the freshly built extension. Real Stockfish/KataGo smoke tests are opt-in and
require provisioned external artifacts.

## Interactive chess deployment

The native interactive chess engine is shared by both deployments and is intentionally retained production code:

- [web play](documentation/operations/web-play.md) uses the typed FastAPI backend and browser client;
- [Lichess/Vast](deployment/lichess/README.md) invokes `python -m src.games.chess.uci` through the checked-in UCI
  launcher.

## Research and references

- [Experiment backlog](THINGS_TO_TRY.md)
- [Research references](documentation/references.md)
- [Historical insights](documentation/history/insights-and-recommendations.md)
