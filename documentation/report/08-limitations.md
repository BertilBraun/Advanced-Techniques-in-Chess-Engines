# 8. What the evidence does not establish

The complete run establishes the strength of the assembled system, not an Elo contribution for every retained
ingredient. Several component comparisons used one seed, a short horizon, frozen replay, or a local proxy. They
establish mechanics and screened choices, but not long-run online gains. Historical policy-head and global-context
comparisons without complete artifacts remain explicitly qualitative. The larger network reached parity during a
limited continuation; that does not distinguish optimization difficulty from a capacity or target-quality limit.

The final ratings come from 100-game paired matches against fixed-node Stockfish 13 and a historical node-to-Elo
calibration. They are benchmark Elo, not FIDE ratings or current unrestricted-engine rankings. Match sampling and
anchor calibration both matter; the displayed bootstrap intervals include only the former. Opponent rungs disagree
at shallow budgets. Policy-only and searched play also use different inference artifacts, while deep-search
parallelism varies with budget. Figure 7.2 is thus a measured operating curve, not an isolated search-scaling law.

The rejected methods' economics depend on this workload. Exact graph reuse and inference-cache hits were too rare
in diverse chess self-play; adaptive stopping saved simulations but barely shortened training under actor/trainer
overlap. A repetitive analysis service or non-overlapped learner might behave differently. The final student was
not matched to the teacher's deployment backend at every budget. Finally, the $43.20 figure covers only 60 hours
of accepted-lineage rental time; discarded work, evaluation, distillation, and total project spending remain
outside it. Reproducibility means a recorded recipe, artifacts, match protocol, and statistical agreement—not
bitwise-identical future weights across proprietary drivers and TensorRT versions.
