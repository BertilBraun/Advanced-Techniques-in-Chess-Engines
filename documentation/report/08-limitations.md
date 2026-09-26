# 9. Limitations and open questions

The study demonstrates an integrated training recipe under a limited compute budget. Its principal limitations
concern attribution of the gains, calibration of playing strength, and transfer to larger models or different
workloads.

## Attribution and scaling

Most component experiments used a single seed, a short continuation, or frozen replay. These controls supported
design selection at affordable cost, but do not identify an independent final-strength contribution for each
component. Policy-head and global-context comparisons also include qualitative observations rather than complete
matched evaluations.

The attribution problem is strongest for coupled choices. Progressive sizing changes capacity and training
history as well as self-play cost; replay growth and reuse alter both data exposure and demand for fresh games.
Matched-compute experiments are needed to separate those effects.

## Persistent plateaus and the remaining engine gap

The remaining gap to leading engines is estimated at roughly 400 Elo. Closing it requires more than improving
the implementation of the existing recipe. Across the project's training runs, playing strength repeatedly
approached a plateau and did not escape it during extended training. These observations make additional hours
under unchanged settings an unpromising route to a substantial gain.

Four changes are the most plausible next tests: substantially larger models to increase representational
capacity; more search during self-play to improve targets; lower replay reuse to increase fresh-game exposure;
and a larger replay window to retain more diverse experience. Each changes a different potential limit on
learning, and each also changes compute demand or data age. They should be compared by strength gained at matched
compute, not solely by training loss or update count.

The limited larger-model continuation reached parity without demonstrating a higher plateau. It therefore leaves
capacity as an open hypothesis rather than ruling it out.

## Rating calibration and evaluation scope

The benchmark scale derives from a published calibration of fixed-node Stockfish 13, rather than direct matches
against unrestricted engines. Each final match contains 100 games, and the reported intervals capture match
sampling uncertainty with the calibration anchors held fixed. Agreement between opponent rungs is close at the
deepest budget and weaker at shallow budgets. A larger match set and broader opponent field would help distinguish
sampling variation from calibration and matchup effects.

The search-budget curve combines deeper search with increased parallelism. It measures the deployed operating
points, not the isolated effect of search depth. Backend differences likewise enter policy-only, searched, and
student evaluations. [Appendix B](appendix-b-evaluation-tables.md) records these settings for matched replication.

## Workload dependence and development cost

The rejected optimizations remain plausible under other workloads. Diverse chess self-play produced too few
exact graph or inference-cache hits to offset lookup and synchronization costs; repeated analysis may offer more
reuse. Similarly, actor-trainer overlap limited the cadence gain from adaptive stopping. An inference-bound
training regime could obtain a larger benefit from the same reduction in simulations.

The $43.20 rental cost covers the final training run, excluding development experiments and evaluation. It
characterizes the cost of executing the recipe rather than discovering it. Reproduction also depends on hardware
contention, drivers, and TensorRT versions; the configuration, published model, and evaluation protocol define
the comparison, while exact weights and timing may vary across executions.
