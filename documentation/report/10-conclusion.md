# 10. Conclusion

In 2.5 days on one eight-GPU node, an AlphaZero-style chess system trained from random initialization produced a
6.3-million-parameter model measuring **3,251 benchmark Elo at 100,000 searches per move** against a fixed-node
Stockfish 13 ladder. Without search, the same model measured **1,658 benchmark Elo**.

The system combines a compact structured policy head and shared convolutional backbone with progressive
small-to-medium sizing, targeted restart states, and prioritized replay. An INT8-capable architecture, native
search, batched TensorRT inference, and distributed training supplied the required volume of self-play data and
optimizer updates. The result demonstrates the strength achievable by this integrated recipe within the available
compute budget.

The component studies also identify limits to local optimization. Better deep-policy fidelity from learned search
allocation did not produce better online learning, and fewer simulations from adaptive stopping yielded little
reduction in cycle time. Exact graph and neural-cache reuse did not offset their overhead. More seriously, removing
searched endgames corrupted value targets, successful TensorRT refits altered predictions, and loss-based promotion
selected a weaker player. These findings connect efficiency to the quality of the data and model actually consumed
by self-play.

Further scaling remains open. The larger model recovered its parent's strength without surpassing it during the
available continuation, while distillation retained substantial strength in a much smaller network without matching
the teacher. Continued training may yield modest gains, but the observed plateaus point toward changes to the
recipe for a substantial improvement. Larger models, deeper
self-play search, lower reuse, and broader replay are the next hypotheses to test against measured strength per
unit of compute.
