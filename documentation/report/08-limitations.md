# 9. Limitations and open questions

The results establish the strength of the integrated system, but leave open which changes contributed most and
how far the recipe can scale. This chapter examines the limits of the experiments, the rating calibration, and
the directions that remain unresolved.

## Which changes caused the gains?

The final run shows what the assembled system achieved, but it does not separate the contribution of every
ingredient. Component experiments often used one seed, short continuations, or frozen replay to make comparisons
affordable. They can explain why a design was chosen without showing how much Elo it added to the final player.
The historical policy-head and global-context comparisons are less complete still: some conclusions rely on
qualitative observations because their original results were not preserved.

This matters most for choices that change several parts of learning at once. Progressive sizing improves early
self-play throughput but changes the network's capacity and training history. Replay growth, reuse, and fresh-game
supply also interact. Longer matched-compute comparisons would be needed to separate their effects. Likewise, the
larger model reached parity during its limited continuation, leaving open whether more training, different targets,
or another initialization would let it use its additional capacity.

## How should the ratings be interpreted?

The reported Elo values put the matches on a common benchmark scale using a published calibration of Stockfish 13
node limits. They are not direct measurements against unrestricted engines. Each final match contains 100 games,
and the intervals quantify uncertainty from those games; uncertainty in the historical calibration is additional.
The two opponent rungs agree closely at the deepest model budget but disagree more at shallow budgets. More games
and a wider opponent field would help establish how much of that discrepancy comes from sampling or calibration.

The strength curve also reflects the settings used to make each search budget practical. Deeper points use more
parallel leaves, so the curve measures the resulting player rather than varying depth alone. Policy-only play and
searched play use different inference backends, as do the compact student and teacher. Appendix B records these
evaluation settings so comparisons can be repeated on the same basis.

## Where might the conclusions change?

Several negative results depend on the workload. Diverse chess self-play offered few exact graph or cache hits;
repeated analysis of the same positions could make reuse more valuable. Adaptive stopping saved search without
greatly shortening a training cycle because search overlapped the optimizer. A learner that spends most of its time
waiting for inference could benefit more from the same saving.

The $43.20 cost describes the final training run, not the experimentation needed to develop it. Reproducing the
recipe is therefore different from repeating the entire project. Hardware contention, drivers, and TensorRT
versions can also change throughput. The preserved configuration, model, and evaluation protocol provide the
starting point for a repeatable comparison, rather than a promise of identical weights or timing on another node.
