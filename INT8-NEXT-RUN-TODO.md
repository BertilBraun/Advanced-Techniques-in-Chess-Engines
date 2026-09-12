# INT8 next-run tracker

Updated: 2026-09-12

## Decision target

Decide whether the measured TensorRT gain, QAT quality, and native-runtime readiness justify another self-play run. Keep GPU probes to 5–10 minutes unless an item below explicitly records a longer matched screen already in progress.

## In progress

- [ ] **Monitor the production transition.** `vast-chess-8gpu-integrated-v35-int8` launched from revision `7f7aa6dc` and resolved configuration SHA `20d66135…4149b`. Verify generation-zero TorchScript bootstrap, generation-2 fold and TensorRT transition, post-fold training, all eight self-play devices, and the first 30-minute Elo evaluations.

## Pending decisions

- [ ] **Decide later model scaling.** The current run starts with fixed 14×160. A larger model requires a deliberate continuation and its own QAT/TensorRT templates; decide from smoothed Elo and trainer/actor utilization.

## Completed evidence

- [x] **Explain the original 3x result.** Original v34 INT8 reached about 183.6k positions/s because TensorRT retained a contiguous INT8 trunk and fused all residual second convolutions, but model fidelity failed catastrophically.
- [x] **Measure the viable production-size INT8 graph.** Scaled post-activation 14x160 reached about 134.6–135.1k INT8 positions/s versus 59.4–60.2k TorchScript BF16 and 97.8–98.0k TensorRT FP16: about 2.24–2.27x core over TorchScript and 1.38x over TensorRT FP16.
- [x] **Test preactivation QAT learning.** Two matched 100k-step seeds differed from floating point by only about 0.025% mean held-out loss, but TensorRT execution was invalid and the architecture is rejected.
- [x] **Test shared residual scales.** The graph was runtime-faithful but slower than TensorRT FP16 at production size, so it is rejected.
- [x] **Demonstrate TensorRT refit.** A seed-13 engine accepted all 160 refittable weights, including 56 Q/DQ constants, and refit to seed 14 in about 0.156 seconds with fresh-engine-equivalent outputs and throughput. Cached rebuilding took about 6.39 seconds versus about 67 seconds uncached.
- [x] **Compile native TensorRT FP16 inference.** The first parity smoke matched all top actions on five legal positions; maximum legal-policy probability difference was 0.00601 and maximum WDL-component difference was 0.00171.
- [x] **Measure native search throughput.** In the production 400-root, 64-visit, parallel-searches-4 loop, TorchScript BF16 sustained about 41.0k simulations/s and TensorRT FP16 about 77.3k simulations/s, a 1.885x gain at matched average batch occupancy.
- [x] **Measure a native INT8 candidate.** The 14x160 early-fold QAT candidate sustained about 98.7k simulations/s versus 75.5k for its TensorRT FP16 form and 41.0k for the v34 TorchScript reference. The 2.405x cross-architecture number is promising but requires the lifecycle smoke and strength validation.
- [x] **Validate native checkpoint refresh.** Two workers atomically refreshed generation 1784 to 1785 in about 0.22 seconds and continued at roughly 72.8k–73.4k simulations/s.
- [x] **Validate repeated TensorRT refits.** All 160 weights, including 56 Q/DQ constants, remained refittable with zero missing weights; a representative refit took about 0.156 seconds and matched a fresh engine.
- [x] **Complete the scaled post-activation replicate.** Two production-size seeds produced stable throughput and fidelity; the viable graph is about 2.24–2.27x faster than TorchScript in isolated inference.
- [x] **Implement the persisted QAT phase model.** Checkpoints distinguish pre-fold and deployment phases, rebuild DDP and optimizer at the fold boundary, and normalize compiled checkpoint keys. Python 3.10 compatibility was restored in `c2338a22`; clean end-to-end validation is still pending above.
- [x] **Complete native TensorRT and QAT lifecycle validation.** Fresh bootstrap, fold, native INT8 refresh, post-fold optimization, checkpoint resume, and batch-64 Stockfish evaluation passed. The QAT resume importer now preserves ONNX artifact identity.
- [x] **Finalize production configuration.** Fixed 14×160 scaled-post INT8, NAG with a 1,000-step warm-up and global `0.1→0.01` decay, replay ratio 6.25, staged 15M replay capacity, policy/value weights 1/1, v34 visits/openings, and 30-minute policy-only plus 64-search Elo evaluation.
- [x] **Launch the production run.** Revision `7f7aa6dccf289521ccf047e2b3e84d36cbb20ab9`; configuration SHA `20d6613593afacbac8ee85e263e41050c838958061ca9bc4d5af623a5154149b`.

## Known risks

- The current C++ production path was TorchScript-only before this work; TensorRT lifecycle bugs may surface only under concurrent self-play and checkpoint refresh.
- The 100k QAT evidence uses a 12x128 model. The 14x160 production-size evidence is currently a short 1k-step smoke, not a full trained model.
- Late BatchNorm folding changes the quantization problem. Five thousand deployment-form recovery steps repaired most fold-specific loss; ten thousand steps regressed.
- Folding replaces convolution modules and introduces bias parameters. Performing it inside live DDP would leave DDP and optimizer references stale, so it must be a persisted phase boundary with a rebuilt optimizer and an intentional momentum reset.
- TensorRT FP16 is already a lower-risk fallback with essentially exact outputs and roughly 1.65x isolated-core throughput on v34.
