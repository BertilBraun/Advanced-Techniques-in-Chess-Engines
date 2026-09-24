# 2. How claims are measured

The evidence required depends on the claim. Faster inference establishes a local throughput gain, and lower policy
loss shows better fit to a given target. Neither demonstrates stronger play. In online self-play, concurrent training
can absorb saved search time, while a network that fits stored positions may produce a worse distribution of new
games. We therefore separate playing strength, learning progress, throughput, and numerical fidelity throughout the
report.

## Playing strength

The terminal chess evaluation plays the selected checkpoint against Stockfish 13 at fixed node limits. Openings are
paired: each starting position is played once from each colour. The protocol fixes the opening suite, opponent
version and resources, candidate artifact, search budget, and parallelism; Appendix B reports the node limit,
W/D/L count, score, and interval for each terminal match row.

The opponent nodes are assigned historical ratings from Marco Meloni's fixed-node Stockfish calibration [9]. The
result is **benchmark Elo** under this protocol, not a FIDE or online rating or a ranking against unrestricted
contemporary engines. At each model search budget, the headline uses the tested opponent whose score is nearest
0.5, limiting extrapolation; both tested limits remain visible. Their implied ratings disagree most at shallow
model budgets, and the available matches do not isolate why. The reported bootstrap intervals quantify match
sampling while holding the calibration anchors fixed, so they do not include calibration uncertainty.

The policy-only row selects legal moves directly from a float TorchScript policy export. Searched rows use the INT8
TensorRT deployment artifact. This distinction is part of the result, not an assumption of backend equivalence.
Search parallelism also changes along the deep-search curve; the curve describes achieved operating points, not a
controlled fixed-parallelism scaling law.

## Learning and comparison

To compare online learning interventions, we measure playing strength over time while matching initial weights,
replay contents, evaluation, and hardware workload as closely as the experiment permits. Independent from-scratch
trajectories can vary more than a small proposed effect; the decisive adaptive-search comparisons therefore started
from the same model and replay state. Frozen-replay fits screen optimizers, policy heads, and quantization schemes,
but cannot establish online self-play strength by themselves.

The cross-campaign 64-search plot is descriptive. The previous baseline and final recipe used different ladder
estimators, so subtracting their plotted endpoints would not be a valid strength comparison. A retrospective
three-rung comparison gives an approximately 74-Elo plateau difference, subject to an approximately ±15-Elo
estimator-transfer sensitivity. That sensitivity is not a game-level confidence interval; Appendix B gives the
comparison boundary and transfer assumption.

## Throughput and target fidelity

Throughput has several distinct units: model evaluations, search simulations, completed games, materialized replay
positions, optimizer steps, and strength gained per wall-clock hour. A gain at one stage may vanish at the next.
Rate claims therefore specify the relevant batch size, concurrent games, GPU contention, search parallelism, and
training overlap. Saturated many-game throughput is not interactive single-game latency.

Serving artifacts must be checked as chess models, not merely as files that load. The TensorRT refit investigation
showed that successful export and refit calls could still yield incorrect legal-move probabilities. Fidelity checks
therefore compare outputs on real encoded positions, including legal-action masking, policy agreement and
Kullback–Leibler divergence, and value error. A float checkpoint's match cannot substitute for a match by the INT8
artifact intended for deployment.

## Provenance and limits

The living final configuration [10] explains the current recipe; it is not by itself a frozen record of the run
reported here. Reproduction also requires the source revision, resolved configuration, selected checkpoint and
deployment artifacts, evaluation assets, and archived results. Appendix D distinguishes public artifacts from
locally archived evidence. A proposal or configuration records intent; implemented code and preserved measurements
are needed to establish that an experiment happened.

Many choices in the final recipe changed together. We call a choice *retained* when it appears there, and attribute
an isolated effect only where a controlled comparison supports one. Missing historical policy-head results and
qualitative owner recollections are identified rather than converted into measurements. The methods and results
needed for each conclusion are stated in the paper; project records provide provenance, not missing premises.
