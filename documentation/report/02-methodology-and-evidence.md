# 2. How claims are measured

The unit of evidence depends on the claim. Faster inference can support a throughput conclusion. Lower policy loss
can show that a target is learnable. Neither by itself shows that the player became stronger. This distinction is
especially important in a self-play system: a saved simulation can be absorbed by concurrent training, and a
network that fits stored targets can still create a worse game distribution.

## Playing strength

The project evaluates chess models against Stockfish 13 at fixed node limits. Openings are paired: each starting
position is played once from each colour. Every reported match identifies the candidate checkpoint and deployed
inference artifact, candidate search budget and parallelism, opponent limit, opening suite, W/D/L count, score, and
uncertainty interval. The terminal matrix and its raw evidence are in the
[final result record](../results/final-chess-run.md#terminal-evaluation-protocol).

Absolute ratings use [Marco Meloni's fixed-node Stockfish calibration](https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/).
They are **benchmark Elo** under this match protocol. They are not FIDE ratings, online ratings, or ratings against
unrestricted contemporary engines. For each model search budget, the headline uses the opponent limit whose match
score is closest to 0.5, reducing Elo extrapolation. Both tested limits remain visible. Their inferred ratings
disagree most at the shallowest model budget; the data do not identify a single cause for that disagreement.

The policy-only row uses direct masked-policy action selection through a float TorchScript export. Searched rows use
the INT8 TensorRT deployment artifact. Those rows answer different questions and retain their artifact identities.
The deep-search curve also changes search parallelism, so it is an operating curve rather than a controlled
fixed-parallelism scaling law.

## Learning and comparison

An online learning comparison should measure playing strength over time while keeping initial model weights, replay
contents, evaluation, and hardware workload as close as possible. When an intervention is small, independent
from-scratch trajectories can differ more than the proposed effect. The decisive adaptive-search comparisons
therefore forked the same model and replay state. A frozen-replay fit is useful for screening an optimizer, policy
head, or quantization scheme, but it cannot establish self-play Elo by itself.

The cross-campaign 64-search plot shows descriptive training trajectories. The previous baseline and final recipe
used different ladder estimators, so the report does not subtract their plotted endpoints as a strength claim. A
retrospective comparison on a matched three-rung estimator estimates an approximately 74-Elo final plateau gain,
with a separate transfer sensitivity of about ±15 Elo. The latter is not a game-level confidence interval. The
[derivation and trimmed plot input](../results/final-chess-run.md#training-trajectory-and-excluded-work) record the
comparison boundary.

## Throughput and target fidelity

The report names each throughput stage: model evaluations, search simulations, completed games, admitted replay
positions, optimizer steps, and Elo per wall-clock hour. A gain at one stage may shrink or disappear at the next.
Batch size, concurrent games, GPU contention, search parallelism, and training overlap therefore belong with any
rate claim. Saturated many-game service time must not be called interactive single-game latency.

Serving artifacts are checked as chess models, not merely as valid files. The TensorRT failure investigation showed
that successful export and refit calls could still produce incorrect legal move probabilities. Fidelity checks use
real encoded positions, legal-action masking, policy agreement and KL, and value error. A float checkpoint's match
cannot stand in for a different INT8 artifact when INT8 is the intended deployment.

## Provenance and limits

The living [final configuration](../../py/configs/production/chess-final-config.yaml) explains the current recipe.
The reported run additionally needs its frozen source revision, resolved configuration, checkpoint and deployment
hashes, evaluation assets, and archived results. The [evidence index](../evidence/final-chess-20260923/README.md)
links those identities. A plan records intent; only implemented code and preserved measurements support claims that
an experiment happened.

Many choices in the successful recipe were changed together. The report calls them *retained* when they are in that
recipe, and gives an isolated effect only when a controlled comparison supports one. Missing historical policy-head
artifacts and qualitative owner recollections are stated as such. The
[experiment ledger](../experiments/README.md) and [coverage matrix](coverage-matrix.md) retain the full audit trail
without making the reader traverse it to understand the main result.
