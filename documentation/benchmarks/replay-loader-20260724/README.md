# Mature rolling-replay loader benchmark

This benchmark validates the Stage 4 disk-backed replay implementation at source
revision `8e6d68d61ec072db957e3131af9f6e1b5bc2b070`.

## Fixture and command

The fixture contains 2,500,000 unique positions in 5,000 producer shards of 500
positions each. The shard size models ten-game producer writes: the v5 iteration-319
replay contained 97,487 positions from 2,283 games, or 42.53 retained positions per
game, rounded conservatively to 50 positions per game.

The benchmark runs two complete 50-step quanta with a global batch size of 1,024 and
four coordinated loader processes. It measures the same replay before and after idle
compaction into 25 immutable containers of 100,000 positions.

```text
PYTHONPATH=/workspace/chess/source/py /venv/main/bin/python \
  tools/benchmark_replay_loader.py \
  --workspace /workspace/chess/artifacts/replay-loader-8e6d68d/workspace \
  --output /workspace/chess/artifacts/replay-loader-8e6d68d/results.json \
  --trainer-consumption-samples-per-second 22599.1267
```

## Results

| Layout | Throughput | Payload opens | Row read amplification |
|---|---:|---:|---:|
| 5,000 producer shards | 3,942.53 samples/s | 36,988 | 70.59x |
| 25 compacted containers | 32,578.80 samples/s | 200 | 194.64x |

The compacted loader is 8.26x faster than the uncompacted layout and exceeds the
recorded 22,599.13 samples/s trainer-consumption rate by 44.16%. Compacting the full
2.5-million-position fixture took 34.29 seconds. Each rank's decoded 12,800-position
quantum occupied 191,628,800 bytes.

Compaction deliberately preserves logical source-shard ranges, so index metadata
remains proportional to producer-shard count: 4,160,904 bytes before compaction and
3,049,579 bytes afterward.

## Limitation

The current grouped reader opens each physical payload once but reads the contiguous
span between its lowest and highest sampled rows. Larger containers therefore increase
read amplification even though they eliminate most file-open overhead. The measured
throughput clears the current trainer requirement, so more selective HDF5 reads are a
follow-up optimization rather than a launch blocker.

Raw measurements are in [results.json](results.json).
