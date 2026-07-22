# Local interactive-engine smoke benchmark (2026-07-21)

This is a reproducibility and regression sample, not a production-strength result. It used a
deterministic one-block, 16-channel-width network whose parameters were all zero. Its uniform
policy and zero value make move agreement useful for detecting parallel-search drift, but say
nothing about chess strength.

The workload was the initial chess position on WSL2, CPU-only PyTorch 2.7.1, four logical CPUs,
one-second timed searches, and 64-search deterministic runs. The reference was a 256-search
single-thread run. Exact commands, model hash, git state, runtime, and hardware are in
`manifest.json`; all candidates and inference statistics are in `results.jsonl`.

Timed highlights:

| Threads | Max batch | Searches | Searches/s | Avg model batch | Overshoot | Reference move |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1 | 1,088 | 1,088 | 1.000 | 0 ms | yes |
| 1 | 2 | 849 | 849 | 1.000 | 0 ms | yes |
| 2 | 1 | 1,880 | 1,873 | 1.000 | 4 ms | no |
| 2 | 2 | 2,188 | 2,186 | 1.107 | 1 ms | yes |

The sample demonstrates why throughput alone is insufficient: two threads with batch size one
completed 73% more searches than the sequential baseline but selected another move under a
uniform policy because parallel selection and virtual loss changed the visit distribution. The
two-thread, batch-size-two configuration was fastest, retained the reference move, and honored
the one-second deadline within 1 ms on this run. These observations need repeated runs and a real
trained model before guiding production configuration.
