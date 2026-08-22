# Naive Python chess MCTS baseline: one RTX 3060

## Result

A deliberately naive, single-game Python implementation sustained a median **80.84 MCTS simulations/s** while
sharing GPU 0 with the live eight-GPU chess training run. Each measured simulation produced exactly one root visit,
one batch-one model inference call, and one inferred position. The three fresh-root runs were:

| Run | Simulations | Root visits | Model calls / positions | Elapsed | Simulations/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 1,000 | 1,000 / 1,000 | 12.389 s | 80.716 |
| 2 | 1,000 | 1,000 | 1,000 / 1,000 | 12.371 s | 80.837 |
| 3 | 1,000 | 1,000 | 1,000 / 1,000 | 12.080 s | 82.782 |

The median is 80.837 simulations/s; the pooled rate is 3,000 / 36.840 = 81.434 simulations/s. This is an
approximate concurrent-load baseline, not isolated GPU capacity. GPU utilization was already high and variable, so
the result should not be quoted with more precision than approximately **81 simulations/s**.

Raw structured output is retained in [`result.json`](result.json).

## Reference implementation

[`py/tools/benchmark_naive_python_mcts.py`](../../../py/tools/benchmark_naive_python_mcts.py) is intentionally
small and direct:

- `python-chess` owns board state, legal moves, terminal detection, and all tree traversal;
- one PUCT tree is searched from the standard initial position;
- selection uses `-Q(child) + c * P(child) * sqrt(N(parent)) / (1 + N(child))`;
- one leaf is selected, expanded, and backed up at a time;
- the TorchScript network receives exactly one position per call;
- CUDA is synchronized after every inference so asynchronous launches cannot inflate the rate;
- there is no game batching, leaf batching, inference queue, cache, tree parallelism, virtual loss, prefetch, or
  compiled native search.

The reference uses the existing native `ChessPosition` only as a model-boundary adapter for the exact 29-plane
encoding and 1,880-action policy mapping expected by the trained model. Search, position copying, legal-move
enumeration, game termination, PUCT, expansion, and backup remain in Python. Constructing the adapter from FEN for
every leaf is itself deliberately unoptimized. It also means repetition-history input planes are not preserved
beyond what FEN contains, although `python-chess` still uses its move stack for terminal draw claims. This benchmark
measures throughput, not move quality.

The timed interval excludes model loading, 32 warm-up simulations, and the one inference call used to initialize
each measured root. It includes Python board copying, selection, FEN conversion, encoding, host-to-device transfer,
batch-one network inference, legal-policy extraction, expansion, and backup. All 3,000 measured leaves were
nonterminal, so simulation and inference-call counts happen to be identical in this workload; they are distinct
metrics and would diverge when a simulation reaches a terminal leaf.

## Model, runtime, and command

The benchmark used the live production chess network rather than a synthetic model:

| Item | Value |
| --- | --- |
| Training run | `vast-chess-8gpu-1d-r4` |
| Runtime source revision | `91924202b7074230b2046a068578a3a9f1ad7951` |
| Checkpoint | generation 481, `model_481.jit.pt` |
| Inference model SHA-256 | `fc725986e2b6eed041f9f27ec3fa4fc8b30e9acd20fc20a586eee85487f64033` |
| Benchmark script SHA-256 | `cd31f6f68f1b025150bc6844bf1953ac83aeeb756f495650d51d82b8fd97a7fa` |
| Raw result SHA-256 | `0725b420cede985e61d72c50599983dcb97ccc98dcdf6fa370d53dabe9b164e5` |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB; physical GPU 0 only |
| Driver / Torch runtime | 580.126.20 / PyTorch 2.12.1+cu126 |
| CUDA / cuDNN | CUDA 12.6 / cuDNN 9.10.2 |
| Python / python-chess | Python 3.12.3 / python-chess 1.11.2 |
| Host CPU visibility | 64 logical AMD EPYC 7452 CPUs |
| Search widths | one game, one leaf, inference batch one |

The exact command, with `CUDA_VISIBLE_DEVICES=0`, was:

```bash
cd /workspace/chess-run-source/vast-chess-8gpu-1d-r4-91924202/py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /workspace/alphazero-engine-venv/bin/python /tmp/benchmark_naive_python_mcts.py \
  --model /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-1d-r4/model_481.jit.pt \
  --output /tmp/naive-python-mcts-generation481.json \
  --device cuda:0 \
  --simulations 1000 \
  --repeats 3 \
  --warmup-simulations 32 \
  --source-revision 91924202b7074230b2046a068578a3a9f1ad7951
```

The script was copied to `/tmp`; no source worktree, run artifact, configuration, approval, replay, checkpoint,
supervisor definition, or training process was modified. The script content is preserved by the repository commit
containing this report rather than by the older runtime revision named above.

## Interference audit

`/etc/vast-agents-guide.md` was read completely before acting on the node. The live process was inspected before and
after through read-only supervisor, process, log, checkpoint, and `nvidia-smi` queries.

| Audit point | Training state | GPU 0 snapshot |
| --- | --- | --- |
| Initial audit, 09:46 UTC | supervisor `RUNNING`, PID 2418017; generation 481 latest | 2,048 MiB, 100%, 145.25 W |
| Immediate pre-run, 09:53:05 UTC | same supervisor PID; generation 481 latest | 2,048 MiB, 53%, 85.19 W |
| Post-run, 09:53:50 UTC | same supervisor PID; generation 482 latest | 2,048 MiB, 100%, 153.21 W |

Generation 482 completed at 09:53:05 UTC with 9,457 training samples/s and the normal credit/replay telemetry. The
benchmark did not stop, restart, signal, or reconfigure any live process. The GPU snapshots are instantaneous, not
averages; together they show why the result must be labelled contention-affected. The run may have reduced
training/self-play throughput briefly, as explicitly accepted for this approximate measurement, but the available
telemetry cannot quantify that transient cost.

## Comparison hooks

The native figures below use different positions, model artifacts, schedules, concurrency, and measurement
lifecycle. They are useful scale indicators, not controlled implementation speedups:

| Workload | Native throughput evidence | Per-GPU-normalized gap from ~80.84/s |
| --- | ---: | ---: |
| Eight-GPU integrated training contention | 169,157 searches/s aggregate | about 262x (`169157 / 8 / 80.84`) |
| Eight-GPU selected 2x512 native topology | 217,874 searches/s aggregate | about 337x (`217874 / 8 / 80.84`) |

The integrated figure is documented in the
[`chess-training-throughput-rtx3060-20260812`](../chess-training-throughput-rtx3060-20260812/README.md) report. The
selected topology is documented in the
[`chess-self-play-latency-rtx3060-20260812`](../chess-self-play-latency-rtx3060-20260812/README.md) report and architecture
ledger. An older native, single-root, sequential CUDA measurement reached approximately 194-258 searches/s, but it
used a synthetic model, a different node/runtime, and different timing windows; see
[`interactive-engine.md`](../harnesses/interactive-engine.md).

The defensible report-level conclusion is therefore:

> A straightforward Python, batch-one AlphaZero search produced only about 81 new leaf evaluations per second on
> an RTX 3060 under live-training contention. The production system reaches hundreds of times more throughput per
> GPU by supplying thousands of independent games to dense native batched inference. Exact attribution requires a
> controlled same-model, same-position, isolated comparison, but the order-of-magnitude scale gap is unambiguous.

This baseline supports the project's central narrative: optimization was not incidental. Reinforcement learning
needed enough self-play positions to learn chess within a four-day commodity-GPU budget, so search organization and
inference density determined whether the experiment was feasible at all.

## Validation

- `ruff format py/tools/benchmark_naive_python_mcts.py py/test/test_benchmark_naive_python_mcts.py`
- `ruff check --fix py/tools/benchmark_naive_python_mcts.py py/test/test_benchmark_naive_python_mcts.py`
- remote locked runtime: `python -m pytest --import-mode=importlib test/test_benchmark_naive_python_mcts.py -q`
  (`3 passed`)
- remote two-simulation CUDA smoke against the generation-481 model before the measured run

The focused tests cover parent-perspective PUCT selection, alternating-perspective backup, and terminal outcome
conversion. Local pytest collection cannot load `AlphaZeroCpp` because the Windows checkout has no local native
extension; the same tests passed in the node's locked Release runtime.
