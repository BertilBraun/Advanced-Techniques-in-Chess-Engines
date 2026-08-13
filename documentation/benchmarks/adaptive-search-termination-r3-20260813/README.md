# R3 adaptive search termination audit

## Decision

**Add instrumentation and rerun the study; do not implement adaptive termination from these records.**

The live R3 sample contains a real opportunity signal: final visit policies are much sharper than in the early-R3
and R2 controls, and a deliberately optimistic final-share extrapolation suggests that a 75%-minimum rule could
remove about 15.3% of nominal fast-search visit limits. With full searches assigned zero savings, that is 6.38% of
all nominal visit limits in this sample. Neither number is an observed saving. Completed games contain no temporal
visit history, no starting retained visits, and no actual simulations performed per root. Full-search records contain
the forced-playout-pruned training target rather than the complete raw child visit vector. The first safe trigger,
leader stability, policy-target divergence, and actual compute saved therefore cannot be reconstructed.

The training-target risk is material. Stopping a full search changes the policy target even when its top move is
already certain. Stopping a fast search also changes any next-policy auxiliary target that consumes it. Value-based
rules are not supported: positions with decisive final root values were less policy-concentrated than close positions.

## Data and selection

At 2026-08-13 16:51:36 UTC, one shell expansion selected every atomically completed `*.json` then present in the
live R3 inbox. A low-CPU/idle-I/O-priority `tar` streamed those files directly to the local machine in 14 seconds;
no remote snapshot file was created. This is a point-in-time census of one accumulation window, not a random sample.
It can overrepresent games that finish within that window and contains games spanning model transitions.

| Sample | Games | Searches | Full | Fast | Generations | Terminations |
|---|---:|---:|---:|---:|---:|---|
| live `vast-chess-8gpu-1d-r3` | 634 | 43,572 | 11,323 | 32,249 | 67–70 | 338 natural, 71 maximum-ply, 225 resignation |
| early R3 2x512 control | 663 | 53,224 | 42,216 | 11,008 | 7–11 | 574 natural, 89 maximum-ply |
| stopped R2 control | 126 | 16,224 | 4,204 | 12,020 | 25–70 | 126 natural |

Live budgets were exactly 500 for full searches and 125 for fast searches. The live games had 97.1 mean plies,
105 median, P10 33, P90 160, and P95 160. Their final WDL labels were 521 losses and 113 wins from the final-position
player's perspective. The controls are unmatched leftover-inbox censuses with different budgets and schedules; they
show historical scale and direction, not causal network-strength effects.

Local tar evidence (raw games are intentionally not committed):

| Sample | Tar bytes | SHA-256 |
|---|---:|---|
| live R3 | 32,471,040 | `3B6A55028BCED9F412958ED76F767C5038354A18D7579FDF4F03D8C4283FF366` |
| early R3 | 51,804,160 | `64453B1475569DF0DA1C81AD196B5D48096139704A57B6D20526509B66E09D10` |
| R2 | 12,021,760 | `4F4764859C494D8FCF3BE7B1BB9052E78501751EEF14BBB232091EA21643EA5C` |

## What records identify

Schema 3 records final policy-target visits, final root value, final raw highest-visited action/count/Q, selected
action, full/fast flag, nominal visit limit, generation, ply, trajectory, outcome, and termination. Fast searches do
not use forced playouts, so their final policy visits are the raw visit distribution. Full searches do use forced
playouts, so their stored policy visits are the pruned training target; only the raw top action/count survives.

Exact replayable measurements are final concentration, final target/raw-top agreement, selected/final-top
agreement, root/top-child value agreement, and game outcomes. Exact stopping time, late overtakes, partial-policy
quality, and simulations saved are unidentified. Even the nominal budget is a visit limit, not work performed:
retained trees may start a search with visits already present.

## Final-policy opportunity

The table reports final target top-one mass. Parentheses are two-sided 95% Wilson intervals.

| Sample/search | ≥0.70 | ≥0.80 | ≥0.90 |
|---|---:|---:|---:|
| live fast | 23.49% (23.03–23.95) | 16.10% (15.70–16.50) | 9.95% (9.63–10.29) |
| live full | 27.68% (26.86–28.51) | 18.10% (17.41–18.82) | 9.32% (8.80–9.87) |
| early-R3 fast | 11.87% | 6.75% | 2.78% |
| early-R3 full | 12.85% | 6.19% | 2.43% |
| R2 fast | 9.68% | 5.73% | 3.56% |
| R2 full | 13.27% | 6.95% | 3.19% |

The live distribution is clearly sharper than both controls. Across all live positions, top mass had mean 0.500,
median 0.468, P10 0.177, P25 0.290, P75 0.694, P90 0.895, P95 0.976, and P99 1.0. The top-two normalized margin had
mean 0.337, median 0.235, P10 0.022, P25 0.072, P75 0.554, P90 0.847, P95 0.960, and P99 1.0.

This does not grow monotonically within the narrow live slice. The ≥0.70 rate was 22.27% at generation 68, 26.89%
at generation 69, and 23.59% at generation 70 (generation 67 had only 36 positions). The 75%-minimum fast heuristic
likewise fell from 20.36 nominal visits at generation 69 to 18.45 at generation 70. Resignation starts at generation
70 and changes the observed position mix, so this is not evidence of network regression or monotonic savings growth.

By ply, ≥0.70 final mass was 22.93% at plies 0–39, 28.96% at 40–79, 22.55% at 80–119, and 22.51% at 120+.
Opportunity is not confined to late play. Using root value as phase/difficulty evidence is particularly unsafe:
positions with `abs(root_value) >= 0.70` had only 16.46% ≥0.70 top mass, versus 26.29% for
`abs(root_value) <= 0.30`.

## Savings scenarios, not measurements

For fast searches only, the proxy extrapolates the final raw top-one and top-two shares backward as constant. It
solves for the first projected point at which the leader's visit lead exceeds every remaining visit, then applies a
minimum search fraction. This assumes precisely the temporal stability that the records do not observe.

| Rule | Mean | Median | P10 | P25 | P75 | P90 | P95 | P99 | Mean fast fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| projected uncatchable, minimum 50% | 25.68 | 22 | 2 | 7 | 43 | 57 | 61 | 62 | 20.55% |
| projected uncatchable, minimum 75% | 19.16 | 22 | 2 | 7 | 31 | 31 | 31 | 31 | 15.33% |

Full-search savings are not estimated: the raw runner-up vector and starting retained visits are absent. If full
searches are conservatively assigned zero savings, the 50%-minimum scenario saves 828,260 projected visits, or
8.55% of 9,692,625 summed nominal limits. Its per-search distribution is mean 19.01, median 11, P10/P25 0, P75 35,
P90 54, P95 60, P99 62. The corresponding 75%-minimum scenario saves 617,975 projected visits, or 6.38%; per-search
mean 14.18, median 11, P10/P25 0, and P75–P99 31. These are heuristic sensitivity bounds, not expected production
speedups; scheduler utilization, retained work, and batching can make wall-time savings differ in either direction.

## Risk proxies

- The stored target top matched the final raw top in all 43,572 live searches (95% Wilson upper bound on disagreement
  0.0088%). This says final forced-playout pruning did not change the winner; it says nothing about an earlier winner.
- The played move differed from the final raw top in 18.23% of fast searches (95% CI 17.81–18.65) and 20.48% of full
  searches (19.74–21.23). This is primarily intended temperature sampling and reserved restart actions, not observed
  leader instability. The rate falls from 46.92% at plies 0–39 to 9.60% at 40–79, 0.82% at 80–119, and 0.17% at
  120+, consistent with the staged greedy threshold.
- Final root value and highest-visited-child Q had the same sign in 95.20% of searches (94.99–95.39), with median
  absolute difference 0.033 and P95 0.132. Both were at least 0.70 in magnitude in 33.24%. They are single final
  estimates with no variance or temporal stability and must not be treated as confidence bounds.
- A later overtake rate, early/final action disagreement, and partial/final policy divergence cannot be measured.
  No confidence interval can repair absent temporal observations.

## Training and runtime interactions

- **Policy targets:** full-search early stopping directly lowers and reshapes the primary target. The current stored
  target also applies forced-playout pruning after search; an early raw distribution need not prune to the same target.
- **Next-policy target:** fast searches are excluded as primary replay rows, but their policy can train the next-policy
  auxiliary head of a preceding full-search row. Fast stopping therefore is not target-free.
- **Root-value blending:** generation 67–70 uses a small but nonzero scheduled root-value blend. Earlier values have
  unknown bias/variance and would change value targets.
- **Resignation:** 225 sampled games resigned. Production resignation requires both final root value and top-child Q;
  stopping early could change triggers and corrupt the continuation-game calibration/audit unless those searches are
  explicitly exempt or shadow-completed.
- **Restart states:** candidate eligibility and branching use full-search final policy mass and root value. Early
  stopping would change archive membership and candidate diversity.
- **Mixed scheduler:** full searches are admitted immediately while fast searches are staged to maintain inference
  capacity. Variable completion can help throughput, but it also changes admission timing and batch fill; nominal
  simulations saved cannot be converted directly to wall time.
- **Move selection:** preserving the current temperature distribution requires sampling from the partial visit vector,
  not replacing it with the current leader. The uncatchable-leader rule only guarantees an argmax under its assumptions.

## Required exact study

Run a shadow-only trace study that never changes the move or target. For a bounded random subset of searches, record:

1. starting retained root visits and raw top-k child `(action, visits, value_sum)`;
2. snapshots every 16 or 25 completed simulations, plus every candidate-rule transition;
3. current root visits/value, remaining visit limit, raw leader and runner-up, forced-playout state, and full/fast flag;
4. first trigger for each candidate rule, subsequent leader overtakes, and the final raw and pruned distributions;
5. actual new simulations and elapsed search time, with batch/admission occupancy;
6. partial-versus-final target KL/JS divergence, top-k mass change, selected move under the same RNG state, and final
   game result; and
7. explicit tags for resignation continuation/audit games, restart roots, and auxiliary-target consumers.

Evaluate an exact unrecoverable raw-visit-lead rule first, with minimums such as 50%, 75%, and 90%. Compare it with
concentration and margin gates only after measuring false triggers and target divergence. Full searches, resignation
audit searches, and restart-source searches should remain shadow-completed until target-quality evidence supports
otherwise.

## Reproduction and non-interference audit

Analysis command:

```powershell
python .\py\tools\analyze_adaptive_search_records.py `
  --sample 'live-r3=C:\Users\berti\.codex\analysis-data\adaptive-search-r3-20260813\live-r3' `
  --sample 'r3-2x512-control=C:\Users\berti\.codex\analysis-data\adaptive-search-r3-20260813\r3-2x512' `
  --sample 'r2-control=C:\Users\berti\.codex\analysis-data\adaptive-search-r3-20260813\r2' `
  --output .\documentation\benchmarks\adaptive-search-termination-r3-20260813\summary.json
```

Before collection, supervisor reported `vast-chess-8gpu-1d-r3 RUNNING`, PID 1920602. `nvidia-smi` showed only the
existing production allocation: one approximately 1.32-GiB process and two approximately 306-MiB processes per GPU,
plus the existing 164-MiB process on GPU 0. After collection, supervisor still reported the same PID RUNNING; the
same GPU process IDs/allocation remained, with the small self-play allocations at approximately 308 MiB.

Remote reads were limited to the instance guide, supervisor/process/GPU status, shallow directory/file metadata,
two bounded live-inbox counts separated by ordinary analysis work, immutable archive counts/sizes, and three tar
streams of final `*.json` files. The first per-file live copy attempt lost its race with normal ingestion and read no
files; it was abandoned. No GPU job or benchmark was launched. `replay.bin`, calibration state, and every restart
SQLite database were neither opened nor copied. No production file, process, configuration, signal, lease, or node
state was changed, and no remote temporary file or instrumentation was left behind.
