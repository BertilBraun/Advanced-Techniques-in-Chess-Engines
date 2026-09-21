# Final chess run

> **Status: training in progress.** This page defines the evidence required for the final result. Empty fields are
> deliberate and must not be replaced by live, partial, or estimated values.

The final run uses the recipe written in full in
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). Development and operational
continuations used the V89–V93 lineage, but the readable final configuration is the stable reproduction entry point.
The published result will also retain the exact source revision, resolved configuration, configuration hash, and
archive manifest from the completed run so later edits to the living recipe cannot change the historical experiment.

## Publication gate

The result becomes final only after all of the following are complete:

- the run is stopped cleanly and its evidence archive is fetched and verified;
- the selected terminal checkpoint and inference artifacts are hashed;
- training-volume and wall-clock statistics are derived from the archive;
- every selected evaluation is complete under its frozen protocol;
- plots are generated from archived inputs and identify those inputs;
- every number in this page is traceable to committed compact evidence.

## Run identity

| Field | Final value | Evidence |
| --- | --- | --- |
| Source revision | **Pending** | Fetched archive provenance |
| Resolved configuration SHA-256 | **Pending** | Resolved configuration and manifest |
| Archive filename and SHA-256 | **Pending** | Local verified archive |
| Effective training lineage | V89–V93 continuation lineage | Run manifests and checkpoint history |
| Canonical readable recipe | [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) | Repository |
| Selected checkpoint generation | **Pending** | Terminal evaluation manifest |
| Checkpoint and inference hashes | **Pending** | Terminal checkpoint manifest |
| Hardware and locked runtime | **Pending final verification** | Archived hardware/runtime provenance |
| Start, stop, and effective training time | **Pending** | Run outcome and telemetry |
| Training-node cost | **Pending** | Frozen node price and effective duration |

## Training volume

| Metric | Final value | Definition |
| --- | ---: | --- |
| Optimizer steps | **Pending** | Completed optimizer steps at the selected checkpoint |
| Training presentations | **Pending** | Optimizer steps multiplied by global batch size |
| Completed self-play games | **Pending** | Accepted completed games through the selected checkpoint |
| Materialized positions | **Pending** | Fresh positions admitted to replay |
| Final replay occupancy and capacity | **Pending** | Live rows and configured logical capacity |
| Effective replay reuse | **Pending** | Training presentations divided by admitted positions |
| Time in each model stage | **Pending** | Effective elapsed time by active progressive model |
| Time in each search-budget stage | **Pending** | Effective elapsed time by configured visit limit |

## Selected model

| Field | Final value |
| --- | --- |
| Progressive stage and architecture | **Pending** |
| Inference parameter count | **Pending** |
| Training precision | **Pending final manifest confirmation** |
| Self-play inference backend and precision | **Pending final manifest confirmation** |

## Terminal strength

The exact opponent nodes, opening count, paired-game count, search parallelism, hardware, and confidence interval
must accompany every row. Search counts and elapsed time are not interchangeable; a time-based headline must also
state the measured serving topology and latency distribution.

| Model search per move | Direct opponent | Games | W/D/L | Score | Benchmark Elo (95% CI) | Status |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| Policy only | **Pending protocol** | **Pending** | **Pending** | **Pending** | **Pending** | Not run |
| 64 | **Pending protocol** | **Pending** | **Pending** | **Pending** | **Pending** | Not run |
| 10,000 | **Pending protocol** | **Pending** | **Pending** | **Pending** | **Pending** | Not run |
| High-search / approximately five seconds per move | **Pending protocol** | **Pending** | **Pending** | **Pending** | **Pending** | Not run |

The project reports protocol-specific benchmark Elo calibrated from fixed-node Stockfish 13 results. It is not a
FIDE rating and is not directly comparable with CCRL, online-server, or current unrestricted-engine ratings. The
existing reporting policy is documented in
[What the v34 Elo numbers mean](../analysis/chess-elo-scale-and-reporting-20260911.md); the final evaluation must
either reuse that calibration exactly or document a revised scale.

## Required figures

- benchmark Elo and match score versus effective training time;
- policy, WDL, auxiliary, and total training losses;
- learning rate, gradient norm, and clipped-step fraction;
- optimizer steps, self-play games, and fresh positions versus effective time;
- self-play, inference, replay-materialization, and trainer throughput;
- replay age, capacity, and sampling distributions;
- progressive-model candidate start and promotion events;
- search-budget, backend, resume, and other material lineage transitions.

Every generated figure must name or link its source archive or committed compact table. Resume gaps and effective
training time must be represented explicitly rather than silently joined on wall-clock timestamps.

## Previous verified reference

Until this page passes its publication gate, v34 generation 1465 remains the latest completed public result. Its
terminal evidence is in the
[v34 benchmark](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md), and its training
trajectory is in the
[v34 dynamics report](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md). Those numbers must
not be presented as measurements of the active final run.
