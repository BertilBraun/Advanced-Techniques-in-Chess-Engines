# 10. Conclusion

An AlphaZero-style chess player trained from random initialization can become very strong on a tightly limited
compute budget, provided its entire learning loop is engineered to turn GPU time into useful games, reliable
targets, and measurable improvement. The selected 6.3-million-parameter model reached **3,251 benchmark Elo at
100,000 searches per move** in the project's fixed-node Stockfish 13 protocol, with a **1,658 benchmark-Elo
policy-only** result. A matched-estimator comparison places its 64-search training plateau about **74 Elo** above
the previous baseline. These are protocol-specific measurements, not FIDE ratings or claims against unrestricted
engines.

The result was not produced by one isolated trick. Compact structured policy prediction, a shared convolutional
network, progressive small-to-medium sizing, searched self-play with restart-state and replay selection, and an
INT8-capable architecture all mattered to the assembled recipe. Native tree ownership, batched inference,
TensorRT, and distributed training made the volume of searched data feasible. The report does not infer an
individual Elo contribution where only a component proxy or the final bundle was measured.

The negative results sharpen that conclusion. A learned allocator produced better deep-policy fidelity but worse
online learning. A stopper saved simulations but barely reduced wall-clock cycle time. Exact graph and neural-cache
reuse were too sparse for their overhead in this chess workload. More parallel leaves sped service while spending
some playing strength, especially at shallow budgets. Three failures exposed the coupling that those local metrics
miss: omitting searched endgames poisoned value targets, a mechanically successful TensorRT refit changed model
behavior, and incomparable training losses promoted a much weaker candidate.

The strongest general lesson is to measure the whole loop. For every proposed saving, ask how many additional
reliable positions it creates, how quickly the learner absorbs them, and whether the deployed artifact actually
plays better. The remaining uncertainty is also clear: the larger model recovered its parent's strength without
surpassing it in the available continuation, and the compact student preserved substantial strength without
matching the teacher. Neither outcome closes the question of how much more chess strength the same framework could
gain from longer training, different targets, or more capacity.
