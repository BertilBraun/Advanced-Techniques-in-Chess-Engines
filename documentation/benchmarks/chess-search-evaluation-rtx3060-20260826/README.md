# Chess offline search evaluation — RTX 3060 — 2026-08-26

Complete result of the overnight Family A / Family B study. Sections 1–5 record staging and smoke work of
2026-08-26/27; sections 6–8 record the run `night-20260826T224742Z`, which completed on 2026-08-27. Design and
reasoning live in
[`plan/search-evaluation-plan-20260826.md`](../../plan/search-evaluation-plan-20260826.md).

## 1. Provenance

| Item | Value |
|---|---|
| Branch | `search-evaluations` |
| Tooling revision | `0408bfb7` — all later commits on this branch touch configuration line endings, documentation and evidence only |
| Source revision on the node | the `search-evaluations` branch tip; every result file records the revision it actually ran under |
| Experiment configuration | `py/configs/validation/vast-chess-4day-production-v9.yaml` |
| `experiment_configuration_sha256` | `28eceba8d2ea74f9fecbfd1770009ad538fc00c08de6344e8cabc922b523b50c` |
| Family A arm matrix | `py/configs/evaluation/chess-search-arms-v1.json`, sha256 `7b7d3b2445e18bf36ffd495467837b44e48b1313487bae55a80cb16cf2026f37` |
| Family B stopping grid | `py/configs/evaluation/chess-search-stopping-grid-v1.json`, sha256 `ee587c89152e98afd7d906525df03490a14d6bd6d029587030954e258642c3b4` |

Configuration JSON is stored and hashed with LF endings, so these hashes match a fresh checkout on the node; each
result file also records the hash of the file it actually read.

The extension used on the node is the Release `AlphaZeroCpp.so` built at `e8bec367` and copied into this branch's
checkout. `git diff e8bec367 search-evaluations -- cpp/` is empty, so the binary matches this branch's C++ exactly;
`e8bec367..HEAD` on `master` touched documentation only.

## 2. Frozen checkpoint

Copied read-only from the live production node `38.49.42.120:53893`, run
`vast-chess-4day-production-v9`, rung `chess-cnn-12x128-dense4`.

| Item | Value |
|---|---|
| Generation | **162** |
| `model_162.jit.pt` sha256 | `a61e850264c71c9c4fd38f0f9660b64b6737052b0516245a34b4e79f3f098571` |
| `checkpoint_162.json` sha256 | `2efa1b0d1131a0e50871aec24fb43fad5df23fa5a3927e4c631ab35b32950103` |
| Copied at | 2026-08-26 ~20:55 UTC |
| Staged on evaluation node | `/workspace/search-eval/run-state/`, hashes re-verified after transfer |

The run's rung directory retains only the newest generation and the run root retains a sliding window, so
generation 162 was first copied to `/workspace/search-eval-freeze/` **on the production node** to take it out of the
pruning window, then fetched in resumable 512 KiB chunks (that node's uplink ran at ~20 KiB/s under the live run)
and verified byte-for-byte at both ends. The live run was not stopped, reconfigured or otherwise touched, and
`run_control.sh` was not used on either node.

## 3. Evaluation node

`50.120.65.61:41841`.

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 3060, 12,288 MiB, UUID `GPU-62a8abba-c105-2754-95ed-c23ad33ebd10` |
| Driver | 595.84 |
| CPU | Intel Xeon E5-2690 v4 @ 2.60 GHz, 56 logical |
| RAM | 62 GiB total |
| Disk | 150 GiB overlay, 68 GiB free at staging |
| PyTorch | 2.12.1+cu126, CUDA 12.6, cuDNN 91002 |
| Python | 3.12.3 (`/workspace/alphazero-engine-venv`) |
| Stockfish | `engines/stockfish-13` → `stockfish_13_linux_x64_bmi2`, 880,622 nodes/s on `bench` |
| Openings | `/workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`, 200 pairs |

**This GPU is shared and was not idle.** Other sessions ran an AlphaZero distillation job
(`tools.distill_match`, then `tools.distill_build_dataset`, ~426 MiB) and a Qwen/TTS generation job
(`/workspace/tts-venvs/qwen`, ~10.1 GiB) throughout staging. Nothing belonging to another session was started,
stopped or modified. Every rate in §4 was measured under that contention and is therefore pessimistic. The
fidelity tool itself holds only **290 MiB** of device memory, so the co-tenants constrain compute, not memory.

Working root on the node is `/workspace/search-eval/` — `source/` (this branch), `run-state/` (checkpoint),
`output/`, `env.sh`, `run-night.sh`. It is deliberately separate from `/workspace/alphazero-engine`, which another
session has checked out with uncommitted work.

## 4. Measured rates (staging, contended GPU)

| Measurement | Rate |
|---|---|
| Rollout sampling, 200 roots, parallel 4, batch 256 | 13,667 simulations/s |
| Reference pass, 40 roots, parallel 4, batch 128 | 8,542 simulations/s |
| Reference pass, 200 roots, parallel 1, batch 256 | 9,967 simulations/s |
| Reference pass, 64 roots, parallel 1, batch 128 | 9,973 simulations/s |
| Family A, one arm, 600 visits, 20 games, Stockfish 300 nodes | 3.94 s/game |
| Family A, one arm, 600 visits, Stockfish 3,500 nodes (the real rung) | 7.3 s/game |
| Family A, six concurrent arms, 600 visits | 0.50 games/s aggregate (2.05× one arm) |
| Family A calibration ladder, 6 rungs × 10 games | ≈ 25 min |

Game cost depends strongly on the rung: balanced games at 3,500 nodes run 1.85× longer than the lopsided games at
300 nodes. Derived budgets: Family B full pass 30M simulations ≈ 50 min; Family A 21 arms × 200 games ≈ 4.4 h at
concurrency 6; ≈ 5.5 h for the night.

### 4.1 Calibration result — the rung is known

| Stockfish 13 nodes | Score (10 games) | W/D/L | s/game |
|---|---|---|---|
| 1,000 | 0.750 | 7/1/2 | 6.1 |
| 2,100 | 0.850 | 8/1/1 | 7.7 |
| **3,500** | **0.500** | **3/4/3** | **7.3** |
| 6,500 | 0.050 | 0/1/9 | 4.9 |
| 11,000 | 0.100 | 0/2/8 | 6.0 |
| 20,000 | 0.150 | 1/1/8 | 6.5 |

`closest_stockfish_nodes` = **3,500**, bracket 2,100–3,500. Ten games per rung is coarse (±0.3), so the tail
non-monotonicity is noise. `ladder_elo_fit` is `null` because these node counts are not all present in
`STOCKFISH_FIXED_NODES_ANCHOR_ELO`; only the rung was needed.

## 5. Staging and smoke tests — all passed

Everything below ran end to end on the node at tiny scale, deliberately without consuming the night's budget.
The last row is the decisive one: the exact script staged for tonight ran all three stages to completion.

| Check | Result |
|---|---|
| Extension import, CUDA visible | ok, RTX 3060 |
| Full pytest suite on the node, this branch | 828 passed, 14 failed, 4 skipped |
| Full pytest suite on the node, baseline `e8bec367` | 805 passed, **the same 14 failed**, 4 skipped |
| Family A gauntlet with overrides, 4 games | ok; overrides recorded in `model_search_budget` |
| Family A arm matrix, 3 arms × 8 games, concurrency 3 | ok; paired differences produced |
| Determinism control: two identically configured arms | paired difference exactly 0.000, interval [0.000, 0.000] |
| Override control: FPU `reduced_parent_value` vs `zero` | different score (0.438 vs 0.375) — the knob reaches the native search |
| Family B position sampler, 8 games → 294 positions | ok |
| Family B fidelity, 40 positions, full 35-rule grid | ok |
| Family B self-consistency (`fixed-10000` vs reference) | KL **0.0**, total variation **0.0**, top-1 **1.000** |
| Replay vs native adaptive search, `parallel_searches=1` | **1.000 exact agreement**, 5 rules × 200 positions |
| Replay vs native adaptive search, `parallel_searches=4` | 0.900–1.000, mean difference 0–25 visits |
| Family A calibration ladder, 6 rungs × 10 games | ok; rung 3,500 chosen (§4.1) |
| **Full `run-night.sh` rehearsal end to end** | **ok, exit 0** — Family B, calibration and all 21 arms in 530 s at 2 pairs/arm |

The 14 node failures are **pre-existing and identical at baseline**, so this branch introduces no regression; it
adds 23 passing tests. They are the known node-only failures noted in `CURRENT-STATE.md`
(`test_trainer_group`, `test_experiment_queue_process`, `test_game_contracts`, `test_interactive_engine`,
`test_benchmark_policy_head_variants`, and a path-separator config-hash test).

### 5.1 The one methodological finding

The Family B replay assumes a search stopped at visit *V* equals the prefix of a longer search at *V*. Tested, not
assumed. With `parallel_searches = 1` it is **exact** (1.000 over 200 positions on each of five rules). With
`parallel_searches = 4` it is ~0.90: the native search is bit-repeatable across runs, so this is not noise — with
several descents in flight the trajectory depends on the visit budget itself, so a 10,000-visit trace is not a
strict prefix of a 600-visit run. Disagreements run in both directions, are one to a few observation intervals
wide, and bias mean visits by ≲3%.

The Family B run therefore uses `--parallel-searches 1`. Production self-play uses 4; that difference is a stated
limit of the result, not a defect in it.

### 5.2 Position sample

`/workspace/search-eval/output/positions-g162-v1.json`, sha256
`a9dc7bd33fe4302d2689e1043ebf9ca2330b3fb2690ad2e07844c6346a14abe5`.

3,000 positions sampled uniformly from 25,766 observed across all 200 book openings, 200 games, 100-visit
rollouts, temperature 1.0, seed 20260826. Ply 0–199, median 69. Positions with a single legal move are excluded.

## 6. Family A results — playing strength

Run `night-20260826T224742Z`, revision `584fa684`, Stockfish 13 at **3,500 nodes**, 100 opening pairs (200 games)
per arm, concurrency 6, 2.2 h. Raw: `results/arm-matrix-result.json`.

The 200-game baseline scored **0.580**, above the 0.500 the 10-game calibration predicted but inside its interval
[0.35, 0.65]. The rung is therefore slightly easy; arms above ~0.70 sit where the score scale compresses, so their
Elo equivalents are less reliable than their score differences.

Baseline: cpuct 1.5, reduced-parent FPU 0.2, virtual loss 1.0, discount 0.99, 600 visits, 4 parallel searches.
**97W/38D/65L, score 0.580**, interval [0.517, 0.642].

| Arm | Score | W/D/L | Paired diff | 95% CI on paired diff | Signif. | Elo vs baseline | Wall-clock |
|---|---|---|---|---|---|---|---|
| `visits-1600` | 0.748 | 137/25/38 | **+0.168** | [+0.080, +0.255] | **yes** | +133 | 4,253 s |
| `visits-1000` | 0.662 | 115/35/50 | +0.083 | [−0.005, +0.170] | no | +61 | 3,581 s |
| `discount-1.0` | 0.640 | 110/36/54 | +0.060 | [−0.030, +0.150] | no | +43 | 2,070 s |
| `cpuct-1.0` | 0.600 | 102/36/62 | +0.020 | [−0.060, +0.102] | no | +14 | 1,930 s |
| `fpu-parent-value` | 0.598 | 102/35/63 | +0.018 | [−0.065, +0.102] | no | +13 | 2,149 s |
| `parallel-4-vlw-0.25` | 0.593 | 102/33/65 | +0.013 | [−0.007, +0.035] | no | +9 | 2,050 s |
| `parallel-8-vlw-1.0` | 0.585 | 98/38/64 | +0.005 | [+0.000, +0.013] | no | +4 | **1,455 s** |
| `parallel-16-vlw-1.0` | 0.585 | 97/40/63 | +0.005 | [+0.000, +0.013] | no | +4 | **1,450 s** |
| `parallel-16-vlw-0.5` | 0.583 | 95/43/62 | +0.003 | [−0.015, +0.022] | no | +2 | 1,461 s |
| **`baseline`** | **0.580** | 97/38/65 | — | — | — | — | 2,011 s |
| `parallel-1` | 0.580 | 96/40/64 | +0.000 | [−0.020, +0.020] | no | 0 | 2,504 s |
| `parallel-8-vlw-0.5` | 0.580 | 94/44/62 | +0.000 | [−0.015, +0.015] | no | 0 | 2,041 s |
| `parallel-4-vlw-0.5` | 0.578 | 95/41/64 | −0.003 | [−0.018, +0.013] | no | −2 | 2,060 s |
| `fpu-reduced-0.4` | 0.575 | 98/34/68 | −0.005 | [−0.095, +0.083] | no | −4 | 2,047 s |
| `cpuct-2.0` | 0.552 | 90/41/69 | −0.028 | [−0.113, +0.058] | no | −20 | 2,121 s |
| `cpuct-1.25` | 0.540 | 85/46/69 | −0.040 | [−0.125, +0.048] | no | −28 | 2,060 s |
| `visits-400` | 0.537 | 90/35/75 | −0.043 | [−0.142, +0.058] | no | −30 | 1,497 s |
| `discount-0.98` | 0.532 | 86/41/73 | −0.048 | [−0.140, +0.045] | no | −34 | 2,069 s |
| `fpu-zero` | 0.472 | 71/47/82 | **−0.107** | [−0.205, −0.007] | **yes** | −76 | 2,038 s |
| `cpuct-3.0` | 0.445 | 69/40/91 | **−0.135** | [−0.225, −0.045] | **yes** | −95 | 2,063 s |
| `visits-200` | 0.318 | 42/43/115 | **−0.263** | [−0.345, −0.177] | **yes** | −189 | 712 s |

Four arms separate from the baseline at 95%.

### 6.1 The evaluation search has been understating every checkpoint

**`fpu-zero` is 0.107 worse than the baseline, interval [−0.205, −0.007], about −76 Elo.** Zero FPU is exactly what
`create_evaluation_search` has forced on every evaluation to date (§0 of the plan). Every ladder number this
repository has recorded for any checkpoint was produced by a search handicapped by roughly this much. This is a
defect in the measurement apparatus, not a tuning preference: the evaluation search should inherit the self-play
first-play urgency.

The other half of the same mismatch does **not** matter: `cpuct-1.0`, the value evaluation hard-codes, is
statistically indistinguishable from self-play's 1.5 (+0.020, interval spans zero).

### 6.2 Visits dominate every other axis

Monotone and by far the largest effect: 0.318 → 0.537 → 0.580 → 0.662 → 0.748 for 200 / 400 / 600 / 1,000 / 1,600
visits. Doubling the current 600 to 1,600 is worth about +133 Elo for 2.1× the wall-clock. 200 visits is
catastrophic (−189 Elo), so the early generations of the v9 schedule are searching far below the useful range.

This agrees with Family B, which found target fidelity still improving steeply at 10,000 visits. Both instruments
point the same way, which is the strongest signal in the study.

### 6.3 Parallelism is free throughput

Every virtual-loss × parallel-searches arm lands within ±0.013 of the baseline and none is significant, but
`parallel-8` and `parallel-16` finish in **1,450–1,455 s against the baseline's 2,011 s — 28% less wall-clock at
no measurable strength cost**. `parallel-1` is both slowest (2,504 s) and no stronger.

Note these arms carry far tighter paired intervals (±0.015) than the FPU and cpuct arms (±0.09). That is the
pairing working: changing parallelism perturbs few moves, so the per-opening differences correlate strongly and
the bootstrap tightens. Changing FPU or cpuct changes many moves and the pairing buys less.

### 6.4 Nulls, and what they bound

`cpuct-1.25`, `cpuct-2.0`, `fpu-reduced-0.4`, `fpu-parent-value`, `discount-0.98` and `discount-1.0` all returned
intervals spanning zero. Per §1.3 of the plan these bound the effect at roughly ±30 Elo; they do not show the
settings are irrelevant. `cpuct-3.0` is the one clear boundary: cpuct is safe anywhere in 1.0–2.0 and harmful by
3.0.

`discount-1.0` (the discount switched off) scored +0.060 with an interval spanning zero — no evidence the search
value discount helps, and a weak hint it costs. Consistent with the recovery analysis twice finding no conversion
effect from it, but not on its own a reason to change it.

## 7. Family B results — policy-target fidelity per unit of compute

3,000 positions × 10,000 visits = 30M simulations in 44 min at 11,350 simulations/s, `parallel_searches = 1`.
Raw: `results/fidelity-g162-v1.json`.

**Self-consistency assertion passed exactly**: `fixed-10000` against the reference returns KL 0.0, total variation
0.0, top-1 agreement 1.000.

### 7.1 The fixed compute–fidelity frontier

| Visits | KL | Total variation | Top-1 agreement |
|---|---|---|---|
| 100 | 0.6501 | 0.2941 | 0.627 |
| 200 | 0.4561 | 0.2650 | 0.674 |
| 300 | 0.3904 | 0.2483 | 0.694 |
| 400 | 0.3473 | 0.2356 | 0.705 |
| 500 | 0.3194 | 0.2249 | 0.717 |
| **600 (v9 today)** | **0.2971** | **0.2161** | **0.724** |
| 800 | 0.2649 | 0.2019 | 0.749 |
| 1,200 | 0.2139 | 0.1798 | 0.777 |
| 1,600 | 0.1815 | 0.1621 | 0.798 |
| 2,400 | 0.1343 | 0.1357 | 0.826 |
| 3,200 | 0.1005 | 0.1140 | 0.853 |
| 5,000 | 0.0525 | 0.0784 | 0.898 |
| 8,000 | 0.0106 | 0.0316 | 0.957 |
| 10,000 | 0.0000 | 0.0000 | 1.000 |

Returns have not flattened by 10,000 visits. At the production 600 visits the policy target names the same best
move as its own 10,000-visit reference only **72.4%** of the time.

### 7.2 Adaptive does save visits, but the saving does not pay

The first write-up of this section overstated the result and is corrected here.

**Adaptive reduces visits as designed.** `adaptive-baseline` averaged 513.9 visits against its own 600 cap, a 14%
reduction; 65% of positions ran to the cap and the other 35% stopped at 354 visits on average. That is exactly the
intended mechanism and it is not in question.

What the study measures is whether those saved visits were *well* saved. They were not: the same target quality is
reached by a **flat 466-visit** budget, so adaptive spent 514 to buy what 466 buys uniformly. Every one of the 21
adaptive rules shows this sign.

#### Decomposing the loss

Two different things cause it, and they carry different implications.

The KL-versus-visits curve (§7.1) has strongly diminishing returns, so it is **convex**. By Jensen's inequality any
rule that spreads visits around a mean scores worse than a flat budget at that mean — *even if it allocates
perfectly*. Splitting the observed penalty into that unavoidable part and the remainder attributable to which
positions the rule chose:

| Rule | Mean visits | Observed KL | Flat at same mean | Convexity penalty | Selection penalty |
|---|---|---|---|---|---|
| `adaptive-baseline` | 513.9 | 0.3290 | 0.3163 | +0.0053 | +0.0073 |
| `adaptive-aggressive` | 443.1 | 0.3536 | 0.3353 | +0.0073 | +0.0110 |
| `adaptive-top-two-margin-0.3-0.1` | 487.1 | 0.3405 | 0.3230 | +0.0068 | +0.0107 |
| `adaptive-maximum-visits-1000` | 742.6 | 0.2972 | 0.2742 | +0.0092 | +0.0139 |
| `adaptive-maximum-visits-1600` | 1006.6 | 0.2769 | 0.2386 | +0.0105 | +0.0278 |

The convexity column is the penalty a *randomly chosen* 35% of positions cut to the same depth would incur. Roughly
**40% of the loss is arithmetic, not a flaw in the stopping rule.** The remaining selection penalty is positive
throughout, meaning the rule chose worse-than-random positions to cut short — its stability signal is read over a
short window and is fooled by temporary plateaus, which is a known MCTS failure mode: a line that looks settled at
400 visits is often overturned by 10,000.

The selection column uses the population-average fidelity curve as a stand-in for the curve of the specific
positions that stopped early. That is an approximation, and it is the weakest step in this analysis — see §7.4.

#### Magnitude depends on the metric

In total variation, which needs no probability floor, the production-scale gap is much smaller:

| Rule | Observed TV | Flat at same mean | Difference |
|---|---|---|---|
| `adaptive-baseline` | 0.2258 | 0.2237 | +0.0021 (~1%) |
| `adaptive-aggressive` | 0.2349 | 0.2310 | +0.0039 (~2%) |
| `adaptive-maximum-visits-1600` | 0.2037 | 0.1905 | +0.0133 (~7%) |

So the honest conclusion is narrower than "adaptive loses everywhere":

- **At the production cap of 600 it is roughly a wash** — about 4% worse in KL, about 1% worse in total variation,
  against a flat budget at its own mean.
- **It degrades clearly as the cap rises** — both metrics agree at caps of 1,000 and 1,600, where it loses 19-28%
  of its visits' worth.
- It never wins anywhere in the grid, at any parameter setting.

The case for dropping it is therefore not that it is catastrophic; it is that it delivers **no measurable gain at
production scale while carrying eight configuration parameters**, and it becomes actively harmful in exactly the
direction the visit-schedule evidence (§6.2, §7.1) says the next run should move — upward.

### 7.3 Limits of the Family B result

Measured at `parallel_searches = 1`, where the replay is provably exact (§5.1); production self-play uses 4.
Searches start from a fresh root while production retains 60% of parent visits. Dirichlet noise is off. The
learned search-correction gate was disabled and is not covered by this result. Target fidelity is not a training
outcome: a cheaper search that reaches different positions could still train better, and this cannot see that.

### 7.4 What would settle this properly

The decomposition above infers the selection penalty from population averages. The decisive measurement is a
per-position one, which the current tool does not emit: record each position's fidelity at every fixed budget,
then compute the **best achievable allocation** at a given mean compute. That bounds what *any* stopping rule
could deliver.

- If the oracle allocation cannot beat a flat budget, no stopping rule ever can, and the mechanism is dead on
  arrival for this metric.
- If the oracle beats flat substantially, the mechanism is sound and only these particular stopping criteria are
  bad, which makes better criteria worth designing.

This costs one further reference pass, about 45 minutes. It is not yet run.

## 8. Recommendations for the next run

1. **Fix the evaluation search.** Make `create_evaluation_search` inherit the self-play first-play urgency instead
   of forcing zero (§6.1). This is a correctness fix worth about 76 Elo of measurement bias.
2. **Raise the visit schedule.** Both instruments agree the current 600 is well short (§6.2, §7.1). The strength
   gain is large and the target-fidelity gain is steep; the cost is linear wall-clock, partly offset by (3).
3. **Raise `parallel_searches` to 8.** 28% less wall-clock at no measurable strength cost (§6.3).
4. **Drop the adaptive full-search budget** — provisionally, pending §7.4. It is a wash at the production cap
   and degrades as the cap rises, so it delivers nothing while carrying eight configuration parameters (§7.2).
5. **Leave cpuct at 1.5** — anywhere in 1.0–2.0 is indistinguishable, 3.0 is harmful.
6. **Leave the search value discount alone** pending better evidence; no effect was detected either way.

## 9. Preserved staging evidence

The node is ephemeral, so the artefacts are committed: the run results under `results/`, and the
small staging artefacts that back sections 4.1 and 5.1 under `staging-evidence/`:

- `calibration-ladder-result.json`, `chosen-rung.txt` — the rung determination.
- `val-par1-adaptive-*.json` — the five replay-versus-native validations at `parallel_searches = 1`, each
  reporting `exact_agreement` 1.0 over 200 positions.

The 3,000-position sample is not committed; it is regenerated exactly from the recorded model hash, seed 20260826
and the sampler arguments in §5.2, and its hash is recorded there.

## 10. Reproduction

On the node:

```bash
source /workspace/search-eval/env.sh
bash /workspace/search-eval/run-night.sh
```

`env.sh` pins the interpreter, the extension, Stockfish, the openings, the frozen checkpoint and
`ENGINE_SOURCE_REVISION`. `run-night.sh` runs Family B, then the Family A calibration ladder, then the Family A
arm matrix, and stops on the first failure. Neither is started automatically.
