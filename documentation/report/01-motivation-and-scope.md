# 1. Motivation and scope

## Research question

AlphaZero is conceptually simple but operationally expensive: the learner improves only as quickly as self-play can
generate useful targets, the network can absorb them, and evaluation can resolve small changes in strength. This
project asks what can be achieved when those activities share one rented node of consumer GPUs and are judged by
strength per wall-clock hour rather than by unconstrained scale.

The goal was not to reproduce DeepMind's compute budget or to approach modern unrestricted chess engines. It was to
build the complete loop from scratch, make it fast enough to study, identify which techniques survive controlled
measurement in this regime, and train a clearly superhuman chess model at modest rental cost. The project includes
game rules, encoding, neural inference, Monte Carlo tree search, self-play, replay, distributed training, evaluation,
deployment, and evidence preservation.

## Scope

Chess is the report's primary subject and the final campaign. The same native and Python runtimes also support Go on
7x7 and 9x9 boards. The early [Go 7x7 baseline](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md)
helped validate the shared runtime, external-engine evaluation, replay credit, and multi-GPU training. It did not
receive a comparable terminal campaign, so this report does not imply a final Go result.

The work has three intertwined outputs:

- a playable chess engine and a reproducible training system;
- an empirical ledger of useful, neutral, failed, and unattempted techniques;
- a final compute-constrained training recipe whose readable entry point is
  [`py/configs/production/chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml).

## Why the negative results matter

The project repeatedly found that a locally successful proxy did not guarantee stronger play. Predicted adaptive
budgets improved policy-fidelity-per-search but lost Elo. Learned early stopping reduced search but barely shortened
the critical path because self-play overlapped training. Attention could look competitive until the policy head and
hardware throughput were separated from the trunk. Early INT8 attempts could be fast yet serve numerically invalid
policies. These are not side notes: they explain the final recipe as much as the retained features do.

The report therefore treats an implementation, a proxy improvement, a throughput improvement, and an Elo improvement
as four different claims. A technique is called “retained” only when it appears in the final configuration; a
technique can still be scientifically informative when it was rejected.

## Project phases

The history is easier to understand as phases rather than as a sequence of version numbers:

1. **Platform construction.** Python orchestration was progressively replaced by a native C++ game/search runtime,
   direct batched inference, columnar replay, and persistent distributed training.
2. **Baseline and recovery.** A four-day chess run established a yardstick. Subsequent runtime work regressed learning
   efficiency, leading to forensic comparisons of encoding, search semantics, schedules, and training data.
3. **Search and architecture research.** The project measured attention trunks, policy heads, progressive sizing,
   search parallelism, adaptive budgets, learned stopping, and target construction.
4. **Compression and inference research.** The v34 model was evaluated, distilled, and used to investigate TensorRT,
   FP16, INT8, quantization-aware training, folding, and refitting.
5. **Final recipe development.** Frozen-replay screens and online runs converged on progressive CNNs, a from-to policy
   head, SGD, pre-fold INT8 serving, growing replay, targeted data selection, and fixed-budget search.
6. **Final run and publication.** The final run is active at the time of this draft. Terminal evaluation and archived
   statistics will complete [Chapter 7](07-final-run-results.md).

The older [platform rework ledger](../architecture/platform-rework.md) and
[Python runtime rework](../architecture/python-runtime-rework.md) preserve the detailed engineering chronology. They
are evidence, not a substitute for the current recipe.
