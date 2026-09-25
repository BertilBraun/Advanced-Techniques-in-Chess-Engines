# 2. How claims are measured

Faster inference does not necessarily produce faster learning: concurrent training can absorb saved search time.
Likewise, lower policy loss on stored positions need not produce stronger play or better self-play data. We measure
playing strength, learning progress, throughput, and numerical fidelity separately.

## Playing strength

The terminal chess evaluation plays the selected checkpoint against Stockfish 13 at fixed node limits. Openings are
paired: each starting position is played once from each colour. The protocol fixes the opening suite, opponent
version and resources, candidate artifact, search budget, and parallelism; Appendix B reports the node limit,
W/D/L count, score, and interval for each terminal match row.

The opponent nodes are assigned historical ratings from Marco Meloni's fixed-node Stockfish calibration [9]. At
each model search budget, we use the tested opponent whose score is nearest 0.5, limiting extrapolation, and show
both tested limits. The resulting **benchmark Elo** measures this fixed-node protocol, not FIDE or online strength.
The two opponents' implied ratings disagree most at shallow model budgets; the matches do not isolate why.
Bootstrap intervals cover match sampling, but not uncertainty in the calibration anchors.

The policy-only row selects legal moves from a float TorchScript export; searched rows use the INT8 TensorRT
deployment artifact. Search parallelism also changes along the deep-search curve. Its points describe the achieved
settings rather than a fixed-parallelism scaling law.

## Learning and comparison

To compare online learning interventions, we measure playing strength over time while matching initial weights,
replay contents, evaluation, and hardware workload as closely as the experiment permits. Independent from-scratch
trajectories can vary more than a small proposed effect; the decisive adaptive-search comparisons therefore started
from the same model and replay state. Frozen-replay fits screen optimizers, policy heads, and quantization schemes,
but cannot establish online self-play strength by themselves.

The cross-campaign 64-search plot shows how the campaigns progressed, but its endpoints use different ladder
estimators. For a numerical comparison, we instead use three shared rungs. That retrospective comparison gives an
approximately 74-Elo plateau difference, with an approximately ±15-Elo sensitivity to transferring the estimator
between campaigns. The sensitivity is distinct from match-sampling uncertainty (Appendix B).

## Throughput and target fidelity

Throughput has several distinct units: model evaluations, search simulations, completed games, materialized replay
positions, optimizer steps, and strength gained per wall-clock hour. A gain at one stage may vanish at the next.
Rate claims therefore specify the relevant batch size, concurrent games, GPU contention, search parallelism, and
training overlap. For scale, an earlier TorchScript evaluation on RTX 4070 SUPER averaged 5.31 seconds per
80,000-search position with 50 positions concurrent on each GPU. Later TensorRT INT8 tests improved saturated
search throughput by 1.39--1.86 times, depending on model and workload. That makes roughly five seconds or less
per 100,000-search position a plausible *batched-service* scale, not a measured single-move latency for the final
model.

Serving artifacts must be checked as chess models, not merely as files that load. The TensorRT refit investigation
showed that successful export and refit calls could still yield incorrect legal-move probabilities. Fidelity checks
therefore compare outputs on real encoded positions, including legal-action masking, policy agreement and
Kullback–Leibler divergence, and value error. A float checkpoint's match cannot substitute for a match by the INT8
artifact intended for deployment.

## Provenance and limits

Reproducing the reported run requires its source revision, resolved configuration, selected checkpoint and
deployment artifacts, evaluation assets, and archived results. The living final configuration [10] describes the
recipe but may change; Appendix D identifies the public artifacts and locally archived evidence.

Many recipe choices changed together. We attribute an isolated effect only where a comparison supports one;
otherwise the result belongs to the assembled system. Historical policy-head comparisons without preserved results
and qualitative recollections are identified as such where they arise.
