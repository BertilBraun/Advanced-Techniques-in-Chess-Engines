# Multi-GPU Stockfish gauntlet

`py/tools/run_stockfish_gauntlet.py` evaluates one frozen chess checkpoint against an unrestricted Stockfish
binary at a fixed node count. It uses the production paired openings and match rules while allowing either an exact
model-search count or an elapsed model-search time per move.

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
  --experiment configs/production/vast-chess-8gpu-1d-r4.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-1d-r4 \
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
  --experiment configs/production/vast-chess-8gpu-1d-r4.yaml \
  --run-directory /workspace/chess-experiment-artifacts/py/training_data/production/vast-chess-8gpu-1d-r4 \
  --checkpoint-generation FINAL_GENERATION \
  --opening-manifest /workspace/evaluation-artifacts/chess/chess-stockfish-8moves-v3-openings-v1.json \
  --stockfish-executable /workspace/evaluation-engines/stockfish-13-source/src/stockfish \
  --stockfish-nodes 1000 \
  --model-move-time-seconds 1 \
  --devices 0 1 2 3 4 5 6 7 \
  --output-directory /workspace/chess-evaluation-gauntlets/final-sf13-n1000-model-t1
```

Change the budget to `--model-move-time-seconds 5` for the five-second run. Timed defaults are 64 parallel searches,
two inference workers, batch size 64, and two outstanding batches per worker. These settings should remain fixed
across checkpoints and runs on the same hardware.

## Output and operation

The output directory must not exist. Each completed worker atomically writes `shards/shard-NN.json`; the parent
writes `result.json` only after it validates exact, non-overlapping coverage of every requested paired game. A
failed or interrupted run leaves completed shard evidence in place but is not resumed automatically. Use a fresh
output directory for a deliberate rerun.

Run long gauntlets under the node's supervisor. Preserve `result.json` and the shard directory off the ephemeral
node before stopping or destroying it. For a write-up, report model budget, Stockfish version and node budget,
opening-pair count, GPU model, parallel/inference settings, W/D/L, paired score interval, and—only for timed
runs—the measured searches-per-move distribution.
