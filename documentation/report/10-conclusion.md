# 10. Conclusion

The project demonstrates that a complete AlphaZero-style chess system can reach strong, superhuman play on a modest
rented multi-GPU budget, but it also shows why the algorithmic summary understates the work. End-to-end progress
depended on native batched search, durable replay, persistent distributed training, trustworthy inference export,
matched evaluation, and the discipline to reject attractive ideas when they did not improve wall-clock strength.

The settled design is conservative where evidence demanded it: convolutional trunks rather than a costlier attention
replacement, tree search rather than graph search, fixed per-generation visits rather than learned per-position
budgets, and explicit quantum boundaries rather than an unmeasured fully asynchronous learner. It is ambitious where
the measurements supported complexity: progressive model sizing, a structured from-to policy head, growing and
surprise-weighted replay, restart states, calibrated resignation, auxiliary supervision, and QAT-backed TensorRT INT8
serving.

Several of the most useful lessons are methodological:

- proxy quality is not playing strength;
- saved search is not saved wall-clock when work overlaps;
- architecture quality and serving throughput must be measured separately;
- shared-state forks are substantially more informative than unrelated short runs;
- inference conversion is part of model correctness, not merely deployment;
- a run without a fetched, hashed archive is not durable evidence.

The final campaign selected a 6.3-million-parameter convolutional model. Under the fixed-node Stockfish calibration,
it measured 1,658 benchmark Elo without search and 2,456, 2,925, 3,114, and 3,251 Elo at successively larger search
budgets from 100 to 100,000 searches per move. Those ratings are protocol-specific rather than FIDE or universal
engine ratings, and the search curve mixes parallelism at the deeper points. The complete intervals, paired-game
counts, artifact hashes, and cost boundary are recorded in the
[final result record](../results/final-chess-run.md).

The campaign also bounded the progressive-capacity result. A loss-based gate promoted a larger but much weaker
candidate because unequal replay presentations made training losses incomparable. Function-preserving growth and QAT
recovery brought the larger network back to parity, but it did not improve the strength curve. The reported model
therefore remains the medium network; capacity was not the immediate constraint under this recipe.

The final publication work is specified in the [publication plan](publication-plan.md), and the
[coverage matrix](coverage-matrix.md) keeps supporting, superseded, and primary evidence reviewable rather than
allowing the polished narrative to erase the experimental ledger.
