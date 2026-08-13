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

## Staged-admission follow-up

The 2026-08-13 follow-up used the same stopped-r2 generation-70 model, 600/150 search budgets, 25% full-search
probability, eight RTX 3060 GPUs, one warm-up batch, and a requested 90-second sample. These measurements include
the CUDA completion-event changes from `10893aa0` and `75b30f7c`. They compare only against this follow-up's own
ratio-admission control because the earlier topology table predates those CUDA changes.
The objective was to use known mixed-search budgets to preserve throughput with fewer concurrent games and hence
shorter game-completion latency; full-batch percentage was diagnostic evidence, not the optimization target.

| Admission/topology | Searches/s | Delta vs control | Mean batch | Full batches | GPU utilization | Mean power | Searches/game/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ratio, 2x512, 2 workers x 2 outstanding | 215,318 | - | 49.89 | 43.52% | 92.5% | 131.1 W | 26.28 |
| Capacity fill, 2x512, 2 workers x 2 outstanding | 217,874 | +1.19% | 49.91 | 49.20% | 93.6% | 136.3 W | 26.60 |
| Ratio, 2x640, 2 workers x 2 outstanding | 218,570 | +1.51% | 58.93 | 75.78% | - | - | - |
| Ratio, 2x512, 1 worker x 2 outstanding | 178,078 | -17.30% | 62.97 | 94.02% | - | - | - |

Capacity fill retains the budget-ratio floor, then admits enough additional fast requests to cover the configured
inference capacity after all full requests have been admitted. Waiting fast requests are still released one at a
time when active fast requests complete. The public Python `search(requests)` call, result order, CUDA
synchronization, and terminal/error/statistics lifecycle are unchanged.

For one 512-game process, 25% full searches means 128 full and 384 fast requests in expectation; the realized count
varies because the choice is random. With capacity for 256 outstanding positions, the idealized new-root schedule
starts all 128 full requests plus 128 fast requests. Each fast completion immediately admits one waiter, producing
three successively admitted cohorts of 128 fast requests. In equivalent visit progress, the active population is 256
through 450 visits, after which only the 128 full requests remain until 600 visits. The waiting-fast pool empties
near 300 visits and the last fast cohort finishes near 450 visits. This is a calculation model, not a four-wave
implementation: admission happens per completed fast request, and retained roots begin with unequal visit counts.

Across the four 150-visit intervals, that ideal population supplies seven of eight possible 128-root half-capacity
blocks: `(3 x 256 + 1 x 128) / (4 x 256) = 87.5%`. Expressed over all potential 64-position batch opportunities,
including idle opportunities, this is `7/8 x 64 = 56` positions. It would still produce a reported mean inference
batch of 64 if every submitted call were full, because an idle opportunity creates no model call. The measured
49.91 mean therefore shows that submitted calls themselves were frequently partial; the planned full-search tail
alone cannot explain it.

The small throughput gain and nearly unchanged mean batch size are plausible and do not by themselves indicate an
implementation defect. The policy fills an initial request-count target, not a persistent queue of inference-ready
leaves. Roots temporarily become unavailable while their leaves occupy outstanding slots; retained roots can
already be near or above the fast visit limit; terminal selections consume no model position; and full searches
finish at different times. In particular, this policy replenishes waiting work after a fast search completes, not
after a full search completes, so the active population can fall below the initial capacity target late in a mixed
call. These effects explain why 256 admitted roots do not imply four continuously full 64-position batches.

The one-worker result confirms that maximizing the full-batch percentage is not the same as maximizing useful
throughput: serialization raised full batches to 94.02% but lost 17.30% searches/s. The 2x640 result obtained a much
higher mean batch at only 1.51% more throughput while increasing per-game latency. Production therefore selects
capacity fill with two 512-game processes per GPU: it is the best measured 512-game policy, preserves the lower
in-flight population, and avoids trading materially slower games for a marginal aggregate gain.

Raw evidence is archived as `cuda-integrated-admission-policy-sweep-results.tar.gz`, SHA-256
`e7766b853e6a2c785fc2de3c69a2dc058f18ae4851b09ff4736fc81e50270772`. The ratio and capacity-fill benchmark
revisions were `9663238223706689a980b9ee32f6bf43497c49f6` and
`7da4e58282f6f885e7a3b82e3894369e4262b345`, respectively.
