# 10. Conclusion

In 2.5 days on one eight-GPU node, an AlphaZero-style chess system trained from random initialization produced a
6.3-million-parameter model measuring **3,251 benchmark Elo at 100,000 searches per move** against a fixed-node
Stockfish 13 ladder. Without search, the same selected model measured **1,658 benchmark Elo** under the policy-only
protocol. Those values describe the project's benchmark, not FIDE ratings or matches against unrestricted engines.

The result belongs to the assembled learning loop: compact structured policy prediction, a shared convolutional
network, progressive small-to-medium sizing, searched self-play with restart-state and replay selection, and an
INT8-capable architecture. Native tree ownership, batched inference, TensorRT, and distributed training made the
necessary volume of searched data feasible. The component studies explain the design choices without claiming an
independent Elo gain for each one.

The negative results sharpen that conclusion. A learned allocator produced better deep-policy fidelity but worse
online learning. A stopper saved simulations but barely reduced wall-clock cycle time. Exact graph and neural-cache
reuse were too sparse for their overhead in this chess workload. More parallel leaves sped service while spending
some playing strength, especially at shallow budgets. Three failures exposed the coupling that those local metrics
miss: omitting searched endgames poisoned value targets, a mechanically successful TensorRT refit changed model
behavior, and incomparable training losses promoted a much weaker candidate.

The strongest general lesson is to measure the whole loop. For every proposed saving, ask how many additional
reliable positions it creates, how quickly the learner absorbs them, and whether the deployed artifact plays better.
The larger model recovered its parent's strength without surpassing it in the available continuation, while the
compact student preserved substantial strength without matching the teacher. Longer training, different targets,
and more capacity remain open paths rather than demonstrated improvements.
