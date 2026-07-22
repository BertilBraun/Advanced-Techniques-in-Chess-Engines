# Interactive result processing and pipeline benchmark (2026-07-22)

This benchmark validates two independent interactive-evaluation changes:

1. Direct contiguous-float WDL validation and direct legal `MoveScore` construction, without
   per-position Torch dispatch or the encoded-move decode/legal-list rescan.
2. An experimental configurable second outstanding inference batch per model replica, with
   readiness-first completion across replicas. The tree remains serially owned; inference workers
   never read or mutate tree nodes.

## Provenance and scope

The checked-in `manifest.json` contains the exact command, model SHA-256, source revision, Python,
Torch, operating-system, and hardware details. This local WSL2 run used four logical CPU cores,
Torch 2.7.1 CPU, and the deterministic untrained benchmark model. It is a throughput and structural
correctness experiment, not playing-strength evidence. `results.jsonl` contains all timed and
fixed-4096-search records.

The RTX 3060 endpoint used by the preceding benchmark rejected the available SSH key on this run,
so no new CUDA result is claimed here. The new harness accepts replicas beyond three and
`--outstanding-batches-per-worker 1,2`; the CUDA matrix should be rerun when access returns with
workers 1-8 and batch sizes 32, 50, 64, and at least one larger candidate.

## Component result

Command shape:

```text
./cpp/build/DirectInferenceBenchmark --model cpp/build/interactive-benchmark.jit.pt \
  --mode <direct|processed_direct> --device cpu --batch-size 64 \
  --workers 1 --iterations 200 --slots 2 --seed 0
```

| Mode | Positions/s | Included work |
|---|---:|---|
| `direct` | 32,383 | forward pass and complete policy/WDL copy |
| `processed_direct` | 13,582 | direct work plus legal-move generation, policy filtering/normalization, and WDL conversion |

The component benchmark deliberately processes independently generated boards. Its first legal-move
lookup can therefore be more expensive than the integrated MCTS path, where terminal checking has
already populated the board's legal-move cache. The integrated stage timer is the authoritative
measure of conversion inside search.

## Integrated three-second results

| Replicas | Batch | Outstanding/replica | Searches/s | Result processing | Owner wait | Worker utilization | Deadline error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1 | 14,298 | 50 ms | 2,625 ms | 83.0% | -4 ms |
| 1 | 64 | 1 | 15,176 | 49 ms | 2,673 ms | 86.6% | -17 ms |
| 1 | 32 | 2 | 6,998 | 40 ms | 2,835 ms | 96.4% | -20 ms |
| 1 | 64 | 2 | 11,862 | 49 ms | 2,696 ms | 93.6% | -20 ms |
| 2 | 32 | 1 | 12,831 | 58 ms | 2,567 ms | 69.8% | -8 ms |
| 2 | 64 | 1 | 21,565 | 77 ms | 2,401 ms | 69.7% | +39 ms |
| 2 | 32 | 2 | **30,937** | 122 ms | 2,377 ms | 93.6% | -2 ms |
| 2 | 64 | 2 | 27,276 | 117 ms | 2,266 ms | 84.9% | -12 ms |

The old RTX profile attributed 1.605 seconds for 31,317 positions to result conversion, about
51 microseconds per position. These local integrated runs spend roughly 1-2 microseconds per
position in the same stage. Hardware differs, so this is not a CUDA speedup claim, but it verifies
that dispatcher-heavy conversion no longer dominates the tree owner.

Two outstanding batches are not universally better: they hurt both one-replica configurations,
while improving the two-replica batch-32 run in this matrix. The production default therefore
remains one; two is an explicit benchmark/tuning option pending the CUDA sweep. Fixed-search runs
completed exactly 4,096 searches, and the Python failure-path test with two outstanding batches
verified that exceptions drain all reservations and virtual loss. Move/rank variation in the raw
records is expected from tied uniform priors in the untrained model.
