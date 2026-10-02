# Current state

As of **2026-09-24**. This page separates completed work from evidence that still has to be frozen for publication.

## Since then: the architecture, and one more run (2026-09-29)

An Lc0 teacher diagnostic found what held the lineage at its plateau: the network's construction. Lc0's T1 in this
project's unchanged search plays at least 360 Elo above the final checkpoint at 64 searches, and a student built
like T1 at equal compute, trained from scratch on 47M teacher-labelled positions, beat the final checkpoint by
about 150 Elo where no convolutional variant moved. It serves at 0.4-0.6x the CNN's rate.
[Summary](analysis/plateau-investigation-v100-onwards-20260928.md) and
[measurements](benchmarks/lc0-teacher-diagnostic-rtx3070-20260926/README.md).

The final self-play run with that architecture ([plan](plan/chess-attention-final-run-plan-20260929.md)) ran 52 hours
on 8x RTX 4080 SUPER with AdamW and stopped on 2026-10-02 at generation 896. Its 10x192 network settled near 2,480
ladder Elo at 64 searches, against the convolutional lineage's plateau near 2,360, and at 100,000 searches scored
0.650 against Stockfish 13 at 200,000 nodes: 3,338 [3,293, 3,381], against the selected CNN's 3,251 [3,206, 3,297].
A learning-rate warm restart lowered its training loss but not its strength.
[Results](benchmarks/attention-adamw-final-run-rtx4080s-20261002/README.md). The results below describe the
convolutional lineage and are unchanged by it.

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
training runs and their scheduled matches are complete. The selected-checkpoint training trajectory and narrow
accepted-lineage cost are also reconciled. Remaining publication work concerns wider actor/search and rejection
telemetry where recoverable, optional figures, and editorial review rather than missing strength results.

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
- a replay buffer growing to 20 million rows with policy-surprise sampling;
- randomized openings and difficulty-weighted restart states;
- calibrated resignation with permanent continuation games; and
- next-policy and remaining-game-length auxiliary targets.

This list describes the integrated system. It does not claim that every component has an isolated causal Elo
measurement. The [experiment ledger](experiments/README.md) and
[technical-report source notes](report/source-notes/README.md) separate retained engineering choices, measured
improvements, rejected techniques, and underdetermined experiments.

The living configuration includes the progressive-candidate match evaluation used by its promotion gate. It loads as
a self-contained configuration; publication must still pin the exact resolved configuration and source revision used
for the reported checkpoint rather than treating future edits to the living recipe as historical provenance.

## Verified result summary

| Claim | Status | Evidence or remaining gate |
| --- | --- | --- |
| Selected checkpoint and deployment hashes | **Verified** | [Compact evidence index](evidence/final-chess-20260923/README.md) |
| Teacher evaluation matrix | **Verified** | Ten 100-game rows and the parallelism sweep are checksum-covered in the [evidence index](evidence/final-chess-20260923/README.md) |
| 100,000-search headline: 3,251 [3,206, 3,297] | **Verified** | Second deep anchor agrees within four Elo; both result manifests are captured |
| Accepted-lineage trajectory | **Publication curve complete** | Final recipe: 180 observations through 2.5 days, endpoint 2,372.2, peak 2,407.6; later experiments excluded by scope |
| Cross-campaign improvement | **Retrospective calculation and sensitivity complete** | About +74 Elo on a matched three-rung estimator, with approximately ±15 transfer sensitivity; [standalone derivation](evidence/final-chess-20260923/plateau-comparison.csv) retained |
| First distilled student | **Verified** | 36,621 steps (7.500 replay epochs); 2,683 [2,637, 2,731] at 10,000 searches |
| Longer distilled student | **Verified** | 110,000 steps (22.528 replay epochs); 2,697 [2,640, 2,753] at 10,000 searches and an unbracketed 2,873 [2,819, 2,935] at 100,000 |
| $43.20 training figure | **Narrow derived measure** | 60 accepted-lineage hours at $0.72/h; not total spend |

## Publication work still open

- Reconcile wider actor/search rates, rejected data, and stage timing where the local archive permits it. The
  selected-checkpoint games, positions, presentations, optimizer steps, replay occupancy, and training dynamics are
  already recorded; total project spend is intentionally not claimed.
- Add optional deployment-fidelity or mechanism figures only where they clarify a supported claim.
- Complete the bibliography, editorial, visual, and evidence-link review of the drafted technical report, then
  inspect a rendered edition if one is produced.
- The public model card, aliases, checksum index, and MIT license have been checked. The project owner confirms the
  live site's deployed artifact is current; keep that confirmation distinct from the frozen run hashes.

## Reader path

Use the [documentation index](README.md) for the current system, experiments, results, and report. Use the
[owner review guide](report/source-notes/owner-review-guide.md) for the small set of questions that require project-
owner memory or editorial judgment; the full source dossiers do not require owner review.

Node facts and access details live only in [operations/current-node.md](operations/current-node.md). Nothing on this
page authorizes a launch, stop, rental, or deletion.
