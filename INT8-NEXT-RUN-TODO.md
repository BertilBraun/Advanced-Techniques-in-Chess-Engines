# INT8 next-run tracker

Updated: 2026-09-12

## Decision target

Decide whether the measured TensorRT gain, QAT quality, and native-runtime readiness justify another self-play run. Keep GPU probes to 5–10 minutes unless an item below explicitly records a longer matched screen already in progress.

## In progress

- [ ] **Measure native self-play throughput.** Compare TorchScript BF16 and TensorRT FP16 through the same production search path, configuration, batch behavior, and checkpoint. Record games, evaluated positions, searches, wall time, GPU utilization, and effective positions/s. Owner: native TensorRT worktree.
- [ ] **Finish the scaled post-activation QAT replicate.** Complete seed 20260914 and its untouched-holdout, ONNX, TensorRT-fidelity, and throughput reports. Owner: INT8/QAT worktree.
- [ ] **Validate repeated TensorRT refits.** Refit one engine across multiple distinct weight and Q/DQ-scale updates; require zero missing refit weights and compare against freshly built engines for outputs and throughput. Owner: native TensorRT worktree.
- [ ] **Complete the native TensorRT backend.** Finish production batch handling, engine/context/buffer lifecycle, atomic refresh, error propagation, and native tests. Validate FP16 first, then a qualifying INT8 engine. Owner: native TensorRT worktree.
- [ ] **Define the production QAT lifecycle.** Specify when normalization is folded/frozen, how activation scales are refreshed, what the training checkpoint owns, and how inference artifacts are exported and refitted without a second divergent model. Owner: INT8/QAT worktree.
- [ ] **Finish optimizer calibration.** Record the short AdamW control and NAG learning-rate/gradient-clip probes. The proposed NAG recipe is momentum 0.9, Nesterov enabled, weight decay `1e-4`, and learning rate linearly decayed from 0.1 to 0.01 through generation 1,000. Resolve the gradient-norm cap. Owner: INT8/QAT worktree.

## Pending decisions

- [ ] **Select the inference precision.** Choose among TensorRT FP16 on the existing v34 architecture, scaled post-activation INT8, or no new run. Base this on measured native self-play throughput and fidelity, not isolated core throughput.
- [ ] **Select the residual architecture.** Decide whether the scaled/bounded post-activation architecture's INT8 gain compensates for its deployment complexity and remaining fidelity error. The shared-scale and preactivation TensorRT graphs are currently rejected.
- [ ] **Select AdamW or NAG.** NAG follows the user's preferred long-run prior, but the current five-minute probe learns more slowly than AdamW. Decide whether the expected generalization benefit warrants lower early wall-clock strength.
- [ ] **Set replay economics.** Reassess replay ratio and replay capacity after measuring the real self-play speedup. Candidate direction: spend increased actor throughput on more distinct positions rather than deeper search; do not lower reuse until trainer and actor cadence are measured together.
- [ ] **Finalize the visit schedule.** Current prior remains 300, 400, 500, then 600 visits, with 800 reserved for the late regime. Change it only if measured TensorRT throughput supports a better data/search trade.
- [ ] **Finalize progressive sizing.** Confirm small and medium architectures, QAT warm-up/fold behavior for each size, candidate-start logic, and the exclusion of the large model.
- [ ] **Make the launch decision.** Require a committed revision, resolved config SHA-256, approval file, Release native build, bounded integrated smoke, fetched evidence, and a concrete rollback/restart path.

## Completed evidence

- [x] **Explain the original 3x result.** Original v34 INT8 reached about 183.6k positions/s because TensorRT retained a contiguous INT8 trunk and fused all residual second convolutions, but model fidelity failed catastrophically.
- [x] **Measure the viable production-size INT8 graph.** Scaled post-activation 14x160 reached about 134.6–135.1k INT8 positions/s versus 59.4–60.2k TorchScript BF16 and 97.8–98.0k TensorRT FP16: about 2.24–2.27x core over TorchScript and 1.38x over TensorRT FP16.
- [x] **Test preactivation QAT learning.** Two matched 100k-step seeds differed from floating point by only about 0.025% mean held-out loss, but TensorRT execution was invalid and the architecture is rejected.
- [x] **Test shared residual scales.** The graph was runtime-faithful but slower than TensorRT FP16 at production size, so it is rejected.
- [x] **Demonstrate TensorRT refit.** A seed-13 engine accepted all 160 refittable weights, including 56 Q/DQ constants, and refit to seed 14 in about 0.156 seconds with fresh-engine-equivalent outputs and throughput. Cached rebuilding took about 6.39 seconds versus about 67 seconds uncached.
- [x] **Compile native TensorRT FP16 inference.** The first parity smoke matched all top actions on five legal positions; maximum legal-policy probability difference was 0.00601 and maximum WDL-component difference was 0.00171.

## Known risks

- The current C++ production path was TorchScript-only before this work; TensorRT lifecycle bugs may surface only under concurrent self-play and checkpoint refresh.
- The 100k QAT evidence uses a 12x128 model. The 14x160 production-size evidence is currently a short 1k-step smoke, not a full trained model.
- Late BatchNorm folding changes the quantization problem. Five thousand deployment-form recovery steps repaired most fold-specific loss; ten thousand steps regressed.
- TensorRT FP16 is already a lower-risk fallback with essentially exact outputs and roughly 1.65x isolated-core throughput on v34.
