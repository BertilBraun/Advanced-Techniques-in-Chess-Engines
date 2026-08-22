# Multi-GPU Stockfish gauntlet

`py/tools/run_stockfish_gauntlet.py` evaluates one frozen chess checkpoint against an unrestricted Stockfish
binary at a fixed node count. It uses the production paired openings and match rules while allowing either an exact
model-search count or an elapsed model-search time per move.

The publication-strength workflow uses
`py/reference/chess-elite-2025-11-balanced-4moves-200-v1.tsv`. Its 200 positions produce 400 games because every
position is played once with the candidate as White and once as Black. The old 50-position `8moves_v3` suite remains
the continuity suite for comparisons with training-time results.

## Balanced elite opening suite

The checked-in selection comes from the frozen November 2025 Lichess Elite archive. It retains games where both
players were rated at least 2500, takes the position after four complete moves, deduplicates transpositions, and
rejects positions with unequal material, check, or a terminal result. Candidates are considered in source-frequency
order. Selection requires a Stockfish 18 evaluation within +/-50 centipawns at 250,000 nodes and admits at most four
positions from one ECO code.

The resulting suite contains 200 unique positions across 114 ECO codes. Source frequency ranges from 66 to 6,312
games. Stockfish evaluations range from -34 to +50 centipawns with a +27.1-centipawn mean. Exact source, engine,
filter, frequency, evaluation, and WDL evidence is in
`documentation/benchmarks/chess-stockfish-ladder-8xrtx3060-20260816/chess-elite-2025-11-balanced-4moves-200-v1-report.json`.

Build its native opening manifest in a prepared revision:

```text
python -m tools.build_chess_opening_manifest \
  --experiment configs/production/vast-chess-8gpu-optimal.yaml \
  --selection reference/chess-elite-2025-11-balanced-4moves-200-v1.tsv \
  --stockfish-executable /workspace/alphazero-engine/engines/stockfish \
  --output /workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1.json
```

The selection SHA-256 is `658a650994214be07a9706f744ce07eb9158b48af34a593f95efa9c70a2824dc`.
`py/tools/build_balanced_chess_openings.py` regenerates the TSV and audit report from the downloaded archive, its
extracted PGN, and the pinned Stockfish binary. The generator hashes both source files itself and refuses to
overwrite existing outputs.

## Exploratory ladder

`py/tools/run_stockfish_ladder.py` runs every requested Stockfish node rung with the same deterministic sampled
opening pairs. `--probe-games` must be 10 or 20, corresponding to five or ten complete color-swapped pairs. These
runs locate a bracket only; do not report their intervals as the final Elo result.

Example for a 64-search candidate:

```text
python -m tools.run_stockfish_ladder \
  --experiment configs/production/vast-chess-8gpu-optimal.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-optimal \
  --checkpoint-generation FINAL_GENERATION \
  --opening-manifest /workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1.json \
  --stockfish-executable /workspace/evaluation-engines/stockfish-13-source/src/stockfish \
  --stockfish-node-ladder 1000 2000 3000 5000 \
  --probe-games 20 \
  --opening-selection-seed 20260815 \
  --match-random-seed 20260816 \
  --model-searches 64 \
  --devices 0 1 2 3 4 5 6 7 \
  --output-directory /workspace/chess-evaluation-gauntlets/final-model-s64-ladder
```

`ladder-result.json` records each W/D/L result, the closest rung by score, and an adjacent above/below-50% bracket
when the observations contain one. Run a separate ladder for every materially different candidate search budget.
Timed probes require an explicit `--parallel-searches` value so the separate 64/128/256 throughput and quality
decision is made before the strength experiment.

## Confirmatory 400-game evaluation

After choosing the opponent rung, run all 200 pairs through the gauntlet:

```text
python -m tools.run_stockfish_gauntlet \
  --experiment configs/production/vast-chess-8gpu-optimal.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-optimal \
  --checkpoint-generation FINAL_GENERATION \
  --opening-manifest /workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1.json \
  --stockfish-executable /workspace/evaluation-engines/stockfish-13-source/src/stockfish \
  --stockfish-nodes CHOSEN_NODE_RUNG \
  --all-opening-pairs \
  --match-random-seed 20260816 \
  --model-searches 64 \
  --devices 0 1 2 3 4 5 6 7 \
  --output-directory /workspace/chess-evaluation-gauntlets/final-sf13-chosen-model-s64-400games
```

The full-suite command uses prefix selection because all 200 manifest entries are included. Partial ladder probes
use seeded sampling across the complete manifest, and the selected indices are recorded in every result.

The two budget modes answer different questions:

- `--model-searches` is the hardware-independent algorithmic-strength measurement. Each GPU worker batches every
  live game in its contiguous opening-pair shard. With 50 opening pairs on eight GPUs, the shards contain seven,
  seven, six, six, six, six, six, and six pairs; all eight shards run concurrently.
- `--model-move-time-seconds` is a hardware-specific deployment-strength measurement. Each GPU runs one timed
  analysis stream, while the eight streams run concurrently. The tool refuses to start if `nvidia-smi` reports a
  compute process on any selected GPU. Every move starts from a fresh search root, matching scheduled evaluation
  semantics rather than retaining a tree across moves. The result records actual searches and elapsed milliseconds.

Both modes use one Stockfish process per GPU shard. Every process uses the experiment's configured thread and hash
limits, applies the requested `--stockfish-nodes` value, and never applies `Skill Level` or `UCI_Elo`. The final JSON
records the source revision, tool/model/opening/Stockfish hashes, Stockfish identity, GPU UUID/model/memory/driver,
complete per-game evidence, paired-bootstrap interval, and shard timings.

## Fixed-search examples

Run from `py/` in a prepared revision. This reproduces the candidate side of the scheduled 64-search evaluation
while distributing the 50 pairs over all eight GPUs:

```text
python -m tools.run_stockfish_gauntlet \
  --experiment configs/production/vast-chess-8gpu-optimal.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-optimal \
  --checkpoint-generation FINAL_GENERATION \
  --opening-manifest /workspace/evaluation-artifacts/chess/chess-stockfish-8moves-v3-openings-v1.json \
  --stockfish-executable /workspace/evaluation-engines/stockfish-13-source/src/stockfish \
  --stockfish-nodes 1000 \
  --model-searches 64 \
  --devices 0 1 2 3 4 5 6 7 \
  --output-directory /workspace/chess-evaluation-gauntlets/final-sf13-n1000-model-s64
```

Use the same command with `--model-searches 1000` or `--model-searches 100000` for the larger fixed budgets. The
default fixed-search settings are deliberately the scheduled evaluator's values: one parallel search, one inference
worker, batch size 64, and one outstanding batch. Override these only when defining a separate benchmark; changing
parallel search changes the search execution semantics even when the final visit count is unchanged.

## Timed examples

Stop training and every other CUDA process before a timed run. The one-second command is:

```text
python -m tools.run_stockfish_gauntlet \
  --experiment configs/production/vast-chess-8gpu-optimal.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-optimal \
  --checkpoint-generation FINAL_GENERATION \
  --opening-manifest /workspace/evaluation-artifacts/chess/chess-stockfish-8moves-v3-openings-v1.json \
  --stockfish-executable /workspace/evaluation-engines/stockfish-13-source/src/stockfish \
  --stockfish-nodes 1000 \
  --model-move-time-seconds 1 \
  --devices 0 1 2 3 4 5 6 7 \
  --output-directory /workspace/chess-evaluation-gauntlets/final-sf13-n1000-model-t1
```

Change the budget to `--model-move-time-seconds 5` for the five-second run. Timed gauntlets currently default to 64
parallel searches, two inference workers, batch size 64, and two outstanding batches per worker. For final reporting,
pass the parallel width explicitly after the 64/128/256 empty-GPU benchmark and keep every inference setting fixed
across ladder and confirmatory runs on the same hardware.

## Output and operation

The output directory must not exist. Each completed worker atomically writes `shards/shard-NN.json`; the parent
writes `result.json` only after it validates exact, non-overlapping coverage of every requested paired game. A
failed or interrupted run leaves completed shard evidence in place but is not resumed automatically. Use a fresh
output directory for a deliberate rerun.

Run long gauntlets under the node's supervisor. Preserve `result.json` and the shard directory off the ephemeral
node before stopping or destroying it. For a write-up, report model budget, Stockfish version and node budget,
opening-pair count, GPU model, parallel/inference settings, W/D/L, paired score interval, and—only for timed
runs—the measured searches-per-move distribution.
