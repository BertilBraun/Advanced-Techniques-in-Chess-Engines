# Current state

As of **2026-09-23**. This page separates completed work from evidence that still has to be frozen for publication.

## Project status

The final chess training lineage is complete. The selected model is a 14-block, 160-channel convolutional network
with global context conditioning and a chess from-to attention policy head. It has 6,315,378 parameters and is
reported through its INT8 QAT TensorRT deployment artifact.

The terminal teacher evaluation matrix is also complete. The model reached **3,114 benchmark Elo [3,065, 3,163]**
at 10,000 searches and **3,251 [3,206, 3,297]** at 100,000 searches. These are protocol-specific ratings against
fixed-node Stockfish 13 anchors, not FIDE ratings or ratings from an unrestricted engine list. See the
[final result record](results/final-chess-run.md) for the complete matrix, confidence intervals, search parallelism,
selection rationale, training trajectory, student result, and limitations.

The tail evidence archive now checksum-covers every reported teacher and student result. Both distilled-student
training runs and their scheduled matches are complete. Remaining publication work concerns archive-derived volume
and cost accounting, the non-headline figures, and report prose rather than missing strength results.

## Final recipe and exact reproduction

[`py/configs/production/chess-final-config.yaml`](../py/configs/production/chess-final-config.yaml) is the living
entry point for reproducing the settled recipe. The published result must additionally pin the exact source revision,
resolved configuration, checkpoint, deployment artifacts, and archive hashes that produced the reported model.

The readable recipe uses:

- progressive convolutional model sizing with match-gated candidate promotion;
- global-pooling context and a chess from-to attention policy head;
- SGD with Nesterov momentum and a decaying learning rate;
- pre-fold quantization-aware training and TensorRT INT8 self-play;
- staged search budgets, reduced-parent FPU, forced playouts, and retained trees;
- a replay buffer growing to 20 million rows with policy-surprise weighting;
- randomized openings and difficulty-weighted restart states;
- calibrated resignation with permanent continuation games; and
- next-policy and remaining-game-length auxiliary targets.

This list describes the integrated system. It does not claim that every component has an isolated causal Elo
measurement. The [experiment ledger](experiments/README.md) and
[technical-report source notes](report/source-notes/README.md) separate retained engineering choices, measured
improvements, rejected techniques, and underdetermined experiments.

One reproducibility issue remains open: the living configuration references a progressive-candidate match gate but
currently lacks its corresponding evaluation definition. The completed run used that match-gated workflow, which is
documented in [progressive model sizing](architecture/progressive-model-sizing.md). The configuration must be repaired
and validated before it is presented as a self-contained reproduction snapshot.

## Verified result summary

| Claim | Status | Evidence or remaining gate |
| --- | --- | --- |
| Selected checkpoint and deployment hashes | **Verified** | [Compact evidence index](evidence/final-chess-20260923/README.md) |
| Teacher evaluation matrix | **Verified** | Ten 100-game rows and the parallelism sweep are checksum-covered in the [evidence index](evidence/final-chess-20260923/README.md) |
| 100,000-search headline: 3,251 [3,206, 3,297] | **Verified** | Second deep anchor agrees within four Elo; both result manifests are captured |
| Accepted-lineage trajectory | **Publication curve complete** | Final recipe: 180 observations through 2.5 days, endpoint 2,372.2, peak 2,407.6; later experiments excluded by scope |
| First distilled student | **Verified** | 36,621 steps (7.500 replay epochs); 2,683 [2,637, 2,731] at 10,000 searches |
| Longer distilled student | **Verified** | 110,000 steps (22.528 replay epochs); 2,697 [2,640, 2,753] at 10,000 searches and an unbracketed 2,873 [2,819, 2,935] at 100,000 |
| $43.20 training figure | **Narrow derived measure** | 60 accepted-lineage hours at $0.72/h; not total spend |

## Publication work still open

- Reconcile total games, admitted positions, replay occupancy, discarded compute, and actual end-to-end spend.
- Generate the remaining training-dynamics and deployment-fidelity figures; the cross-campaign ladder figure is
  complete.
- Repair and validate the missing progressive-candidate evaluation definition in the living configuration.
- Replace the previous public result in the root README once the evidence and figures above are frozen.
- Complete the narrative technical report and add explicit code and model licenses.

## Reader path

Use the [documentation index](README.md) for the current system, experiments, results, and report. Use the
[owner review guide](report/source-notes/owner-review-guide.md) for the small set of questions that require project-
owner memory or editorial judgment; the full source dossiers do not require owner review.

Node facts and access details live only in [operations/current-node.md](operations/current-node.md). Nothing on this
page authorizes a launch, stop, rental, or deletion.
