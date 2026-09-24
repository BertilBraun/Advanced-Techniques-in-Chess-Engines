# 10. Conclusion

In 2.5 days on one eight-GPU node, an AlphaZero-style chess system trained from random initialization produced a
6.3-million-parameter model measuring **3,251 benchmark Elo at 100,000 searches per move** against the project's
fixed-node Stockfish 13 ladder. Its **1,658 benchmark-Elo policy-only** result and approximately **74-Elo**
matched-estimator gain over the previous 64-search training baseline describe different protocols. Neither is a
FIDE rating or a claim against unrestricted engines.

The result belongs to the assembled learning loop. It combines compact structured policy prediction, a shared
convolutional network, progressive small-to-medium sizing, searched self-play with restart-state and replay
selection, and an INT8-capable architecture. Native tree ownership, batched inference, TensorRT, and distributed
training made the necessary volume of searched data feasible. Component measurements explain why these choices
were retained, but do not assign each an independent Elo contribution.

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
