# Chess offline search evaluation — RTX 3060 — 2026-08-26

Skeleton for the overnight Family A / Family B study. Sections 1–5 are complete and recorded from the staging and
smoke work of 2026-08-26/27. **Sections 6 and 7 are placeholders**: they are filled in only after the full run,
which has not been started. Design and reasoning live in
[`plan/search-evaluation-plan-20260826.md`](../../plan/search-evaluation-plan-20260826.md).

## 1. Provenance

| Item | Value |
|---|---|
| Branch | `search-evaluations` |
| Source revision on the node | `50d4b076` (tooling code unchanged since `0408bfb7`; later commits touch configuration line endings and documentation only) |
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

## 6. Family A results — NOT RUN

To be filled after the run. Required contents: calibration ladder result and the chosen Stockfish rung; per-arm
wins/draws/losses, score and interval; paired difference against `baseline` with `excludes_zero`; per-arm
wall-clock; and an explicit statement of which axes moved beyond the ±0.037 unpaired detection threshold and which
returned a null that only bounds the effect at ~30 Elo.

Raw output: `arm-matrix-result.json` plus one `result.json` per arm.

## 7. Family B results — NOT RUN

To be filled after the run. Required contents: the fixed-visit compute–fidelity frontier; per-rule mean stop
visits, KL, total variation and top-1 agreement; and the `equal_compute_comparisons` table answering whether
adaptive reduces visits at equal fidelity (`visit_saving`), improves fidelity at equal compute
(`kullback_leibler_advantage`), or neither. The `fixed-10000` self-consistency row must read KL 0.0 / top-1 1.000
or the pass is invalid.

Raw output: `fidelity-g162-v1.json`.

A 200-position staging probe gave the following, recorded **only** to show the analysis path works — n = 200 is far
too small to conclude anything and these numbers must not be quoted as results:

| Rule | Mean visits | KL | Top-1 |
|---|---|---|---|
| `fixed-200` | 200.0 | 0.4523 | 0.670 |
| `fixed-600` | 600.0 | 0.2924 | 0.725 |
| `fixed-1600` | 1600.0 | 0.1881 | 0.815 |
| `fixed-5000` | 5000.0 | 0.0588 | 0.895 |
| `fixed-10000` | 10000.0 | **0.0000** | **1.000** |
| `adaptive-baseline` | 523.5 | 0.2923 | 0.730 |

with `adaptive-baseline` showing `visit_saving` +77.7 and `kullback_leibler_advantage` +0.011 against the fixed
frontier.

## 8. Reproduction

On the node:

```bash
source /workspace/search-eval/env.sh
bash /workspace/search-eval/run-night.sh
```

`env.sh` pins the interpreter, the extension, Stockfish, the openings, the frozen checkpoint and
`ENGINE_SOURCE_REVISION`. `run-night.sh` runs Family B, then the Family A calibration ladder, then the Family A
arm matrix, and stops on the first failure. Neither is started automatically.
