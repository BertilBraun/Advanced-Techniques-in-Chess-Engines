# V39 INT8 self-play throughput investigation

Date: 2026-09-13

## Conclusion

V39's deployment-phase TensorRT INT8 actor is fast. At the production search shape it delivered 205,242 simulations/s on one RTX 4070S, 1.861 times the 110,293 simulations/s of the matched floating TorchScript control. The live run produced 1,365.0 accepted positions/s from generations 20 through 34, 23.3% more than the 1,107.5 positions/s measured for V35 at its later 600-visit topology.

The apparent lack of generation speedup came from translating simulations into replay credit. One V39 training quantum is 500 optimizer steps × 2,048 positions = 1,024,000 presentations. With reuse 6.25, each generation requires 163,840 newly materialized positions. The observed live rate predicts:

```text
163,840 positions / 1,365.037 positions/s = 120.0 s/generation = 30.0 generations/hour
```

That is the observed generation-20-to-34 rate: 14 generations in 1,680 seconds. There is no missing simulation throughput in this interval.

## Reproducibility

- Source revision: `b6b446989dbe7be933189eab5dcf5ea0f6391a6d`
- Experiment configuration SHA-256: `429cf62a75b0ff3c78eb5e97f083014b644b3f390c82e1b274248b97bb653788`
- V39 generation-34 INT8 ONNX SHA-256: `60b3f44a6e753d48524071ac952a884caf3a489a44b8e71f06b115f0c31a4d62`
- Floating control SHA-256: `1c8b5da29c58097a28d2e83af3f73a7c6ad6795a488a348620719165fa65219e`
- Node: `38.49.42.120:53893`; only physical GPUs 6 and 7 were used.
- V39 remained stopped and preserved at generation 34. No production training run was started.

All benchmark arms used the production `SelfPlayWorker` and native search path, 512 concurrent games per process, 400 baseline visits, automatically selected parallel-search count 2, one inference worker, batch cap 320, two outstanding batches, the production opening/restart mix, retained-root fraction, and maximum game length. Raw manifests, worker output, GPU telemetry, and CPU telemetry are under `raw/remote-results`.

The control is the same 12×128 architecture and production configuration, exported from the generation-18 pre-fold checkpoint as floating TorchScript. A byte-identical generation-34 floating control cannot be exported: generation 34 has already folded batch normalization and has a structurally different state dictionary. The control therefore isolates backend/precision with the closest available trained checkpoint, but it is not an identical-weight comparison.

Representative commands were:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHON_BINARY=/workspace/alphazero-engine-venv/bin/python \
GPU_COUNT=1 PROCESSES_PER_GPU=4 PARALLEL_GAMES_PER_PROCESS=512 \
WARMUP_BATCHES=2 MEASUREMENT_DURATION_SECONDS=180 BENCHMARK_GENERATION=34 \
CHECKPOINT_MANIFEST=/workspace/alphazero-engine-int8-validation/py/training_data/production/vast-chess-8gpu-progressive-v39-int8/checkpoint_34.json \
BENCHMARK_OUTPUT_ROOT=/workspace/selfplay-throughput-v39/int8-p4-long \
bash py/tools/run_self_play_search_benchmark.sh \
/workspace/alphazero-engine-int8-validation/py/training_data/production/vast-chess-8gpu-progressive-v39-int8/model_34.int8.onnx \
/workspace/alphazero-engine-int8-validation/py/configs/production/vast-chess-8gpu-progressive-v39-int8.yaml \
/workspace/alphazero-engine-int8-validation
```

The two-process arm changed `CUDA_VISIBLE_DEVICES=7`, `PROCESSES_PER_GPU=2`, and its output root. The floating arm used physical GPU 7, generation 18, `INFERENCE_BACKEND=torchscript`, and the exported floating model. Every remote command was sent through `deployment/remote_command.sh`.

## Benchmark results

| Arm | Duration | Processes/GPU | Sims/s/GPU | Relative | Mean batch | Process CPU | Mean GPU | Mean power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V39 gen34 TensorRT INT8 | 60 s | 4 | 205,242 | 1.861× float | 319.35 | 460.8% | 91.2% | 188.3 W |
| 12×128 gen18 TorchScript BF16 | 60 s | 4 | 110,293 | 1.000× | 319.26 | 242.4% | 93.7% | 172.4 W |
| V39 gen34 TensorRT INT8 | 180 s | 4 | 202,163 | 1.000× p4 | 319.12 | 455.8% | 92.0% | 190.1 W |
| V39 gen34 TensorRT INT8 | 180 s | 2 | 123,369 | 0.610× p4 | 319.01 | 254.6% | 58.0% | 133.8 W |

Four actor processes are required to fill this GPU. Reducing to two processes loses 39.0% of simulations/s and leaves the GPU underused. Both topologies fill inference batches, so the loss is process concurrency around inference rather than batch formation.

A final 30-second p4 instrumentation arm reproduced 206,959 simulations/s and reported 4,836 ns of search-owner inference wait, 4,895 ns of inference-thread work, and 2,059 ns of result processing per completed simulation. The current native backend reported zero for its selection, board-encoding, and backup counters, so those three fields cannot be decomposed further from this build. Together with 92% GPU use and only 4.6 process CPU cores per GPU in the long arm, the measured counters identify inference/concurrency as the exclusive-run limiter rather than host search or serialization.

The standalone completion counters must not be interpreted as steady games/s or positions/s. Every arm starts all 512 games in every process from a fresh empty worker. At 180 seconds, p4 had 2,048 games in flight and reported only 506 completed games; p2 had 1,024 in flight and reported 450. The resulting 90.4 and 99.6 completed positions/s respectively are dominated by cold-start completion and right-censoring. The p2 arm's longer completed games (45.6 versus 37.4 plies) are also selection bias, not evidence that p2 creates more replay data. The 28-minute production ledger is the valid completed/materialized throughput measurement.

## Production decomposition

Between completion of generations 20 and 34, the live coordinator logged:

- 54,802 completed games and 2,293,262 accepted positions in 1,680 seconds.
- 117,433 games/hour, 1,365.037 accepted positions/s, and 41.846 accepted positions/game.
- 404 materialization append batches. Their logged append time totaled 260.9 seconds and their aggregate append rate was 8,789.8 positions/s, 6.44 times the arrival rate. Replay serialization/materialization had ample throughput.
- Inference batches averaged about 318/320 in the live TensorBoard series and 319/320 in the isolated arms.
- Generation 29 through 34 training took about 53–57 seconds, credit wait took 51–57 seconds, and checkpoint publication/refit/activation added about 7–10 seconds.

The current topology pauses 16 of 32 actors during training. Actors therefore run at half population for roughly the 55-second training segment and full population during roughly the 55-second credit-wait segment. This explains why multiplying an exclusive-GPU microbenchmark by eight overpredicts live accepted output: the benchmark has no concurrent trainer, no actor pauses, and no checkpoint activation. It also counts simulations immediately, while replay credit appears only after a full game completes, serializes, passes materialization, and contributes accepted observations.

V39's early policy produced 41.85 accepted observations/game in the measured window. The cited V35 steady measurement implies 59.32 positions/game (`1,107.5 × 3,600 / 67,206`), so a V39 game carried 29.5% fewer replay positions. V39 compensated with 117,433 games/hour, 74.7% more than V35's 67,206 games/hour, and still produced 23.3% more accepted positions/s. Game length and completion gating erase much of the raw simulation gain when viewed as replay credit.

The early-hour comparison also contradicts the impression that V39 was unusually slow. From process start, V39 completed generation 28 at 58:03 and generation 29 at 60:04. V35-from-scratch completed generation 19 at 59:37 and generation 24 only at 69:17. V39 was about nine generations ahead at one hour. Its first generation took 6:42 because all 16,384 games started cold and replay credit requires completed games; V35's first generation took 9:34 for the same reason.

V34 used reuse 8 while V39 uses 6.25. At the same 1,024,000-presentation quantum, V39 consequently requires 28% more new positions per generation: 163,840 rather than 128,000. Holding the measured V39 live position rate fixed, reuse 8 predicts 93.8 seconds/generation or 38.4 generations/hour, a 28% generation-rate increase. This setting changes the learning/data tradeoff; it is not a search optimization.

## QAT folding and evaluation templates

V39 does not run its first 10,000 optimizer steps with unquantized self-play. Its self-play configuration has TensorRT QAT templates for both `pre_fold` and `deployment` phases, and checkpoints publish INT8 inference artifacts in both phases. Folding changes the training graph by absorbing batch-normalization parameters into convolutions after the QAT warmup. TensorRT refit requires an engine template with exactly the same graph structure, so the pre-fold and deployment phases need separate templates. The template is a structural engine shell whose weights are refitted from each checkpoint; it is not a slow preliminary evaluation pass.

Evaluation has separate batch-64 templates because its inference shape differs from self-play's batch-320 shape. Missing or mismatched evaluation templates can delay evaluation startup, but template folding is not what controls actor throughput, and the generation-34 deployment actor result proves post-fold INT8 is active.

## Recommendation

Keep the current self-play topology for the next run: four actor processes per GPU, 512 games/process, batch cap 320, one inference worker/process, two outstanding batches, and automatic parallel searches (2 at 400 visits). Do not reduce to two actors/GPU. Keep the half-actor pause arrangement for training until an overlap benchmark measures a better split; the exclusive p4 arm already uses 92% GPU, so running all actors beside DDP risks contention.

If the goal is more optimizer generations per hour, set replay reuse back to 8. At measured throughput this should move the post-fold rate from about 30 to about 38 generations/hour. Treat replay-capacity reduction as a learning-quality experiment rather than a throughput fix: materialization is 6.4 times faster than arrivals and is not the bottleneck.

The remaining useful instrumentation change is included on this branch: production-path benchmark output now records completed positions/bytes/game length, accepts a real checkpoint manifest and backend override, isolates restart-state storage per worker, and reports native selection, encoding, result, backup, inference-wait, and inference timing counters.
