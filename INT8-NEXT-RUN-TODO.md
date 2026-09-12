# INT8 next-run tracker

Updated: 2026-09-12

## Decision target

Decide whether the measured TensorRT gain, QAT quality, and native-runtime readiness justify another self-play run. Keep GPU probes to 5–10 minutes unless an item below explicitly records a longer matched screen already in progress.

## In progress

- [ ] **Complete the native TensorRT backend.** Finish production batch handling, engine/context/buffer lifecycle, atomic refresh, error propagation, and native tests. Validate FP16 first, then a qualifying INT8 engine. Owner: native TensorRT worktree.
- [ ] **Implement the resumable QAT phase boundary.** After 1,000 optimizer steps, save the pre-fold checkpoint, reconstruct the model in folded deployment topology, recalibrate, rebuild DDP and the optimizer, and persist the phase in checkpoint metadata. The optimizer reset is deliberate because folding replaces parameters. Owner: INT8/QAT and native TensorRT worktrees.
- [ ] **Exercise the full lifecycle.** Run a bounded remote smoke through generation-zero TorchScript bootstrap, pre-fold training, the fold/restart boundary, INT8 publication/refit, native self-play, checkpoint refresh, fixed-dataset and search evaluation, and resume. Fetch its evidence before considering launch. The first two attempts exposed real template and publisher-path failures; a clean run remains required.
- [ ] **Finish optimizer/config integration.** Use NAG with momentum 0.9, Nesterov enabled, weight decay `1e-4`, gradient-norm cap 5, and the selected learning-rate schedule. Resolve replay ratio, head width, and model progression from the measured actor speedup.

## Pending decisions

- [ ] **Select the inference precision.** Choose among TensorRT FP16 on the existing v34 architecture, scaled post-activation INT8, or no new run. Base this on measured native self-play throughput and fidelity, not isolated core throughput.
- [ ] **Select the residual architecture.** Decide whether the scaled/bounded post-activation architecture's INT8 gain compensates for its deployment complexity and remaining fidelity error. The shared-scale and preactivation TensorRT graphs are currently rejected.
- [ ] **Select AdamW or NAG.** NAG follows the user's preferred long-run prior, but the current five-minute probe learns more slowly than AdamW. Decide whether the expected generalization benefit warrants lower early wall-clock strength.
- [ ] **Set replay economics.** Reassess replay ratio and replay capacity after measuring the real self-play speedup. Candidate direction: spend increased actor throughput on more distinct positions rather than deeper search; do not lower reuse until trainer and actor cadence are measured together.
- [ ] **Finalize the visit schedule.** Current prior remains 300, 400, 500, then 600 visits, with 800 reserved for the late regime. Change it only if measured TensorRT throughput supports a better data/search trade.
- [ ] **Finalize progressive sizing.** Confirm small and medium architectures, QAT warm-up/fold behavior for each size, candidate-start logic, and the exclusion of the large model.
- [ ] **Make the launch decision.** Require a committed revision, resolved config SHA-256, approval file, Release native build, bounded integrated smoke, fetched evidence, and a concrete rollback/restart path.
- [ ] **Separate TensorRT publication identities.** Self-play batch-320 and evaluation batch-64 engines must not share `model_N.int8.trt.engine`; include template/input shape or purpose in the cache/output identity and prove concurrent publication cannot swap incompatible engines.
- [ ] **Make evaluation QAT-aware.** Fixed-dataset evaluation currently loads TorchScript directly, while post-fold QAT checkpoints publish ONNX. Generation-zero search evaluation also needs an explicit bootstrap path and all configured evaluation templates must exist.
- [ ] **Remove inherited learning-rate warm-up.** Override the v34 lineage's `warmup_optimizer_steps: 1000`; the selected NAG schedule starts at 0.1 and linearly decays to 0.01 by global generation 1000.

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

## Known risks

- The current C++ production path was TorchScript-only before this work; TensorRT lifecycle bugs may surface only under concurrent self-play and checkpoint refresh.
- The 100k QAT evidence uses a 12x128 model. The 14x160 production-size evidence is currently a short 1k-step smoke, not a full trained model.
- Late BatchNorm folding changes the quantization problem. Five thousand deployment-form recovery steps repaired most fold-specific loss; ten thousand steps regressed.
- Folding replaces convolution modules and introduces bias parameters. Performing it inside live DDP would leave DDP and optimizer references stale, so it must be a persisted phase boundary with a rebuilt optimizer and an intentional momentum reset.
- TensorRT FP16 is already a lower-risk fallback with essentially exact outputs and roughly 1.65x isolated-core throughput on v34.
