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

The final quantitative conclusion is intentionally pending. Once training stops, the archive is verified, and the
terminal evaluation matrix is complete, [Chapter 7](07-final-run-results.md) will state the selected checkpoint,
training volume, cost, policy-only and searched strength, uncertainty, latency, and comparison with v34. The root
README can then present the concise headline while this report preserves the full chain from research question to
evidence.

The final publication work is specified in the [publication plan](publication-plan.md), and the
[coverage matrix](coverage-matrix.md) keeps supporting, superseded, and primary evidence reviewable rather than
allowing the polished narrative to erase the experimental ledger.
