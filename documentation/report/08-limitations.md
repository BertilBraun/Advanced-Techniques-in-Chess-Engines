# 9. What the evidence does not establish

The complete run measures the assembled system, but it cannot assign an Elo gain to each ingredient. Several
component studies used one seed, a short horizon, frozen replay, or a local proxy. They help explain mechanisms and
screen choices, but do not establish long-run online gains. Historical policy-head and global-context comparisons
without complete artifacts remain qualitative. The larger network reached parity in a limited continuation; more
training would be needed to separate optimization difficulty from limits in capacity, replay, or target quality.

The final ratings come from 100-game paired matches against fixed-node Stockfish 13 and a historical node-to-Elo
calibration. The bootstrap intervals cover match sampling, not uncertainty in that calibration. Opponent rungs
disagree at shallow budgets; policy-only and searched play use different inference artifacts; and deep-search
parallelism varies with budget. Figure 9 therefore shows attainable operating points under the stated protocol,
not an isolated search-scaling law or a rating against unrestricted engines.

The rejected methods' economics also depend on workload. Exact graph reuse and inference-cache hits were too rare
in diverse chess self-play; adaptive stopping saved simulations but barely shortened training under actor/trainer
overlap. A repetitive analysis service or non-overlapped learner might behave differently. The student comparison
uses a different deployment backend from the teacher. The $43.20 rental figure covers only the selected training
path, not total project spending. Finally, a reproducible run should preserve its recipe, artifacts, match protocol,
and statistical results; proprietary drivers and TensorRT versions preclude a promise of bitwise-identical weights.
