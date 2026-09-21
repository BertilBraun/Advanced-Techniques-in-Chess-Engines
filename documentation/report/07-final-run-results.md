# 7. Final-run results

> **Status: pending.** The final chess run is still training. This chapter is the only location in the technical
> report for incomplete terminal values. Do not replace pending fields with live dashboard readings or extrapolations.

The project-level result record is [`documentation/results/final-chess-run.md`](../results/final-chess-run.md). Once
the archive and evaluations are complete, that page is the quantitative authority; this chapter should interpret it
rather than duplicate every artifact.

## Frozen run identity

| Field | Final value |
| --- | --- |
| Public run name | **Pending** |
| Training lineage and resume boundaries | **Pending audit** |
| Source revision | **Pending archive** |
| Resolved configuration SHA-256 | **Pending archive** |
| Run-manifest SHA-256 | **Pending archive** |
| Archive SHA-256 and location | **Pending fetch and verification** |
| Start, stop, and effective training duration | **Pending completion** |
| Hardware/runtime identity | **Pending frozen manifest** |
| Training-node cost and exclusions | **Pending completion** |

The lineage audit must distinguish continuous learning state from operational run identifiers V89–V93. Downtime,
failed exports, discarded segments, evaluation compute, and node rental should be reported consistently rather than
compressed into one ambiguous “training time” number.

## Selected model

| Field | Final value |
| --- | --- |
| Checkpoint generation and optimizer step | **Pending selection** |
| Active progressive stage | **Pending** |
| Training/inference parameter counts | **Pending manifest** |
| Checkpoint hash | **Pending** |
| ONNX/TensorRT artifact hashes | **Pending** |
| Precision and serving backend | **Pending frozen artifact** |

Selection should be declared before terminal match interpretation or governed by a pre-stated rule. If a non-terminal
checkpoint is selected, document why and preserve the terminal checkpoint too.

## Training volume

| Measure | Final value |
| --- | --- |
| Completed self-play games | **Pending** |
| Fresh materialized positions | **Pending** |
| Training presentations | **Pending** |
| Optimizer steps | **Pending** |
| Final replay occupancy/capacity | **Pending** |
| Effective replay reuse | **Pending** |
| Time and volume per model stage | **Pending** |
| Time and volume per visit stage | **Pending** |

Counts must come from the fetched archive and reconcile coordinator, replay, and trainer accounting. If restart or
resume boundaries create duplicate counters, report the reconciliation method.

## Terminal evaluation matrix

The final protocol should include policy-only, the production-scale 64-search condition, an intermediate/deep budget
such as 10,000 searches, and a high-search condition chosen to represent roughly five seconds per move under a
documented saturated workload. The exact high budget must be fixed by measurement, not assumed from v34.

| Candidate search | Opponent and limit | Games | W/D/L | Score | Benchmark Elo (95% CI) | Latency |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| Policy only | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** |
| 64 searches | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** |
| 10,000 searches | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** |
| High-search condition | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** | **Pending** |

Every row requires balanced paired openings, exact Stockfish identity and fixed-node limit, search parallelism,
inference batch/concurrency, artifact hashes, raw games, and confidence intervals. The comparison with v34 must use
matched protocols or explicitly identify differences.

## Figures to generate from the archive

1. Total, policy, WDL, and auxiliary losses against wall-clock and optimizer step.
2. Learning rate, gradient norm, and clipping fraction.
3. Policy-only and searched ladder Elo against wall-clock, with uncertainty and model-promotion annotations.
4. Generations, games, fresh positions, and training presentations against wall-clock.
5. Self-play and trainer throughput, including pause and visit-stage changes.
6. Replay occupancy, age distribution, and sampling mixture over time.
7. INT8 legal-policy fidelity over time, with engine-template rebuilds and model transitions.
8. Cost/strength comparison with the v34 result under matched definitions.

Figures must be generated from archived machine-readable evidence, record their source files, and avoid hand-entered
curves.

## Result interpretation to complete

The completed discussion should answer:

- How much stronger was the selected model than v34 under identical search and opponent conditions?
- Did progress continue after reaching v34 strength, and at what marginal Elo per additional wall-clock day?
- Which model and visit transitions changed throughput or learning slope?
- Did policy-only strength and searched strength improve together?
- How much wall-clock and cost were lost to operational faults or resume boundaries?
- Does the deepest measured search continue to add strength, and how far is the result from unrestricted engines?

Until those inputs exist, the abstract, root README, and conclusion should contain a clearly marked result placeholder
rather than a speculative live number.
