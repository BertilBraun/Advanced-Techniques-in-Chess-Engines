# Current state

As of **2026-09-21**. This file distinguishes verified artifacts from live or provisional claims.

## Project status

The final chess training lineage is active. Its effective recipe is written out in full in
[`py/configs/production/chess-final-config.yaml`](../py/configs/production/chess-final-config.yaml), which is the
living reproduction entry point. Operational continuations from V89 through V93 preserve the same run directory and
checkpoint lineage while repairing deployment fidelity and progressive-promotion control. The completed publication
must pin the exact source revision, resolved configuration hash, checkpoint, and fetched archive in addition to
linking the living recipe.

No terminal number from the active lineage is final yet. The required measurements and publication gate are in the
[final-run result record](results/final-chess-run.md).

The latest completed public checkpoint remains v34 generation 1465, retained at about three days of effective
training on 8x RTX 4070 SUPER GPUs. Its rounded training-node cost was **$52**. This excludes separate evaluation
nodes and abandoned or restarted run segments, so it is a training-checkpoint cost rather than a complete project
invoice.

## Verified reference result

| Claim | Status | Evidence or remaining gate |
| --- | --- | --- |
| v34 checkpoint: generation 1465, 14x160, 6,256,365 inference parameters | **Verified** | Frozen teacher manifest in the [compression benchmark](benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| 3,037 benchmark Elo [3,012, 3,061] at 10,000 searches | **Verified** | [400-game terminal benchmark](benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md); `parallel_searches: 4` |
| 3,167 benchmark Elo [3,143, 3,193] at 80,000 searches | **Verified** | [400-game terminal benchmark](benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md); `parallel_searches: 8` |
| 80,000-search batched throughput: 5.31 seconds/move mean | **Verified** | 400 isolated positions across 8x RTX 4070 SUPER; 5.19-second median |
| 474,069-parameter student is 13.20x smaller | **Verified** | Published model and hashes in the [compression benchmark](benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| Student trails by 166.2 Elo under saturated equal expected time | **Verified** | 400 games, 64/186 searches, interval [-200.2, -136.0] |
| Student is superhuman at large search | **Unmeasured hypothesis** | Requires its own calibrated high-search Stockfish match |

Absolute engine Elo here is a Stockfish-node calibration. It is not directly a human FIDE rating or a rating from
another engine list. The repository should keep node count, search configuration, calibration source, match score,
and confidence interval beside every absolute number.

## Active final recipe

The readable final configuration currently specifies:

- progressive 12x128, 14x160, and 19x176 convolutional models with Elo-plateau candidate starts;
- global-pooling context every second residual block and a chess from-to policy head;
- SGD with Nesterov momentum and a linearly decaying learning rate;
- pre-fold quantization-aware training with TensorRT INT8 self-play after generation zero;
- self-play visits staged from 300 through 800, reduced-parent FPU, forced playouts, and retained trees;
- a replay buffer growing to 20 million rows, replay reuse four, and policy-surprise sampling;
- randomized openings and restart states selected from a bounded recent archive;
- calibrated resignation with permanent continuation games;
- next-policy and remaining-game-length auxiliary targets;
- bracketed policy-only and 64-search Stockfish evaluations every 20 minutes.

This is a configuration summary, not evidence that each treatment independently improved strength. The experiment
catalog and technical report distinguish retained engineering choices from causally validated improvements.

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

- Continue the live final lineage until the user explicitly decides to stop it.
- Preserve, fetch, and verify the final archive before releasing the node.
- Derive the final training volume, effective duration, cost, model-stage history, and throughput statistics.
- Complete the terminal policy-only, 64-search, 10,000-search, and selected high-search evaluations.
- Replace the pending fields in the root README and technical report with evidence-backed final results.
- Complete the documentation refactor and formal technical report.
- Add an explicit code and model license. Until then, redistribution terms are unspecified.

## Operational state

Node facts and access details live only in [operations/current-node.md](operations/current-node.md). Use
[`deployment/run_control.sh`](../deployment/run_control.sh) for all run lifecycle actions and
[`deployment/setup_remote.sh`](../deployment/setup_remote.sh) for fresh nodes. Nothing in this document authorizes a
launch, stop, rental, or deletion.

## Reader path

Use the [documentation index](README.md) for current architecture and operations. The old recovery work packages,
v-series decision plans, and adaptive-search plans remain useful as a research ledger, but they no longer describe
the active phase. Use [`chess-final-config.yaml`](../py/configs/production/chess-final-config.yaml) for the readable
current recipe and the future frozen final-result record for the exact published experiment.
