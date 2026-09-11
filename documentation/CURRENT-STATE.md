# Current state

As of **2026-09-11**. This file distinguishes verified artifacts from live or provisional claims.

## Project status

The compute-poor chess training campaign is approaching completion. The main publication checkpoint is v34
generation 1465, retained at about three days of wall-clock training on 8x RTX 4070 SUPER GPUs. Training was allowed
to continue after that snapshot to test the tail, but the three-day checkpoint currently gives the clearest
time/cost/strength result.

The rental cost through that checkpoint was **$52**, rounded from three days at the node's billed rate of $17.36
per day. This excludes separate evaluation nodes and abandoned or restarted run segments, so it is a
training-checkpoint cost rather than a complete project invoice.

## Result status

| Claim | Status | Evidence or remaining gate |
| --- | --- | --- |
| v34 checkpoint: generation 1465, 14x160, 6,256,365 inference parameters | **Verified** | Frozen teacher manifest in the [compression benchmark](benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| 3,037 benchmark Elo [3,012, 3,061] at 10,000 searches | **Verified** | [400-game terminal benchmark](benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md); `parallel_searches: 4` |
| 3,174 benchmark Elo [3,144, 3,206] at 80,000 searches | **Verified** | [400-game terminal benchmark](benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md); `parallel_searches: 8` |
| 474,069-parameter student is 13.20x smaller | **Verified** | Published model and hashes in the [compression benchmark](benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| Student trails by 166.2 Elo under saturated equal expected time | **Verified** | 400 games, 64/186 searches, interval [-200.2, -136.0] |
| Student is superhuman at large search | **Unmeasured hypothesis** | Requires its own calibrated high-search Stockfish match |

Absolute engine Elo here is a Stockfish-node calibration. It is not directly a human FIDE rating or a rating from
another engine list. The repository should keep node count, search configuration, calibration source, match score,
and confidence interval beside every absolute number.

## What v34 tested

The integrated recipe uses:

- self-play visits ramped from 300 to 600, then to 800 late in training;
- search parallelism 2 at 300–400 visits and 4 at 500–800 visits;
- progressive 12x128 to 14x160 model sizing, with promotion triggered from the searched-Elo EMA;
- AdamW with a staged learning rate, including a late drop to 0.001;
- a 10-million-record replay, replay ratio 8, and policy-surprise sampling;
- 50% uniformly randomized 0–8-ply openings and 50% regret-prioritized restart states;
- eight-move history and additional chess feature planes;
- policy loss weight 1.5 and value loss weight 1.0.

The exact resolved configuration and its SHA remain the authority. This list is a reader summary, not a substitute
for the archived configuration.

## Completed closing work

- The replay-compression study is complete and published with the selected student weights, 25 hashed artifacts,
  throughput probes, and three match conditions.
- The generation-1465 deployment checkpoint and compressed student are published in the
  [Hugging Face model repository](https://huggingface.co/BertilBraun/alphazero-chess) with immutable revision,
  source, configuration, architecture, and artifact hashes.
- The interactive chess client and the project homepage report the terminal v34 result with its calibration and
  confidence intervals. The interactive backend is pinned to the generation-1465 production artifacts.
- The [Elo-scale analysis](analysis/chess-elo-scale-and-reporting-20260911.md) defines the public reporting language:
  these are protocol-specific benchmark ratings calibrated from historical SSDF-derived anchors, not FIDE ratings
  or current engine-list ratings.
- The v29 benchmark series and generation-936 deep match provide a fully documented predecessor.
- Adaptive search budgeting and learned early stopping were rejected on Elo evidence and removed from both Python
  and C++.
- The runtime is back to a two-tensor policy/WDL inference contract and fixed visit limits.

## Work still open

- Stop and preserve the final live continuation only on explicit user instruction; fetch and verify its archive.
- Decide whether to write a formal technical report after the repository and public pages are coherent.
- Add an explicit code and model license. Until then, redistribution terms are unspecified.

## Operational state

Node facts and access details live only in [operations/current-node.md](operations/current-node.md). Use
[`deployment/run_control.sh`](../deployment/run_control.sh) for all run lifecycle actions and
[`deployment/setup_remote.sh`](../deployment/setup_remote.sh) for fresh nodes. Nothing in this document authorizes a
launch, stop, rental, or deletion.

## Reader path

Use the [documentation index](README.md) for current architecture and operations. The old recovery work packages,
v-series decision plans, and adaptive-search plans remain useful as a research ledger, but they no longer describe
the active phase.
