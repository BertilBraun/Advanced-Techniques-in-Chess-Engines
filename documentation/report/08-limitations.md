# 8. What the evidence does not establish

The system reached strong play under a limited single-node training budget, but the final recipe combines many
decisions. Search, replay, architecture, quantization, and model sizing changed together. A complete run establishes
the performance of that assembly; it does not assign a separate Elo gain to every retained ingredient. Several
component comparisons used one seed, a short horizon, or frozen replay. Those studies can establish mechanics or
rank a local proxy, but they cannot be promoted into long-run online strength claims.

The terminal ratings are tied to paired games against fixed-node Stockfish 13 and to an external node-to-Elo
calibration. They are benchmark Elo, not FIDE ratings or predictions against current unrestricted engines. The
finite opening suite and 100-game terminal rows leave sampling uncertainty; two opponent rungs also disagree at
some budgets. We show both rungs and select the one closest to a 50% score, but do not claim to know why their
inferred ratings differ. The policy-only and searched points use different serving artifacts, and the deep-search
points use different degrees of parallelism. The plotted curve is a measured operating curve, not an isolated
search-budget scaling law.

The economics of rejected methods belong to this workload. Exact graph-state reuse and inference-cache hits were
too rare to pay their overhead in diverse chess self-play. Adaptive stopping saved simulations but barely shortened
the learning cycle because training overlapped self-play. A repetitive analysis service, another game, or a
non-overlapped learner could change those conclusions. Similarly, the matched policy-head and trunk screens do not
prove that convolution is universally preferable to attention.

Some historical policy-head and global-context comparisons survive only as owner recollection rather than complete
result artifacts. Their qualitative interpretation is separated from preserved measurements in the investigation
chapters. The larger-network continuation reached parity but not a clear gain; its limited duration cannot separate
an optimization problem from a capacity or target-quality limit. The final student experiments show useful
compression, but did not test a student and teacher under identical deployment backends at every budget.

The selected-checkpoint training-volume and trainer-throughput figures are reconciled, but wider end-to-end
self-play/search throughput and total project expenditure are not. The known **$43.20** is only the accepted 60-hour lineage at **$0.72/h**, not the total
cost of experiments, evaluation, or discarded work. Proprietary drivers and TensorRT versions also limit bitwise
reproduction: the intended reproducibility target is a recorded recipe, artifacts, protocol, and statistical
agreement, not identical future weights.
