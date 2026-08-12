# Chess self-play concurrency and completion latency

This benchmark selected the self-play concurrency for `vast-chess-8gpu-1d-r3` on Vast instance `47400225`, using
all eight RTX 3060 GPUs, the stopped r2 run's generation-70 TorchScript checkpoint, and the production generation-70
search mixture. Each isolated search measurement used one warm-up batch and a requested 90-second measurement
window. No trainer or evaluator was active. The primary metric is completed MCTS searches per second; the short
window is not suitable for comparing completed-game counts directly.

| Processes/GPU | Games/process | Active games | Searches/s | Mean inference batch |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 1,024 | 24,576 | 213,401 | 62.03 |
| 2 | 512 | 8,192 | 201,404 | 42.46 |
| 3 | 512 | 12,288 | 198,872 | 42.82 |
| 4 | 512 | 16,384 | 189,320 | 42.96 |
| 3 | 256 | 6,144 | 162,183 | 25.04 |
| 4 | 256 | 8,192 | 158,905 | 24.77 |

The selected topology is two processes per GPU and 512 games per process. It retains 94.38% of the maximum measured
search throughput while reducing the in-flight population by two thirds. The important benefit is completion
latency, not greater aggregate game throughput: the 3x1,024 reference distributes 213,401 searches/s over 24,576
games, or 8.68 searches/game/s, whereas 2x512 distributes 201,404 searches/s over 8,192 games, or 24.59
searches/game/s. Each active game therefore advances approximately 2.83 times faster.

At generation 70, the 25% full-search mixture with budgets of 600 full and 150 fast searches averages 262.5 searches
per searched ply. The topology-only approximate searched-ply latency falls from 30.2 seconds to 10.7 seconds. Under
comparable game lengths, trajectories should finish in roughly 35% of the previous wall time. Scaling the r2 run's
observed approximately 40-minute median completion time gives a rough 14-minute topology-only estimate; early
material termination and calibrated resignation should shorten it further.

Aggregate completion throughput remains about 94.4% of the reference: one-third as many games advance 2.83 times
faster. Thus the topology makes completed trajectories arrive much sooner and reduces generation lag without
increasing games/hour. The 256-game process pools under-fill their isolated inference queues and lose about one
quarter of aggregate throughput. Adding a third or fourth 512-game process does not improve inference occupancy and
adds enough contention to reduce throughput.
