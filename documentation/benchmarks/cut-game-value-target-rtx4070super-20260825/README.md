# Is the search root value a usable terminal-value target for ply-capped chess games?

Measurement, 2026-08-25. Question: at early training generations, when a self-play game is cut at the
ply cap, is the search's own root value at the cut a better value target than
`Board::approximateResultScore()` (raw material / 39)? Is it sign-correct, and how far off is its
magnitude?

**Verdict: bootstrap.** The root value beats the material heuristic on every aggregate metric at both
generations measured, and — the point the concern turned on — it is *well calibrated at generation
16*, not noisy. The material map is the mis-calibrated one. Sharpening the material map is the worse
option: the sharpening factor that is right at generation 30 is badly wrong at generation 16.

## Provenance

| item | value |
| --- | --- |
| node | `38.49.42.120:53893`, 8x NVIDIA GeForce RTX 4070 SUPER, driver 595.71.05, 80 logical CPUs |
| torch | 2.12.1+cu126, CUDA 12.6, cuDNN 91002 |
| source revision on the node | `ca0ef85cd7d528dd089291c7fe6d7db98a844fb5` |
| native extension | Release `AlphaZeroCpp.so` from the v6 production tree, copied to `/workspace/rootvalue-study` |
| configuration | `vast-chess-4day-production-v6`, resolved, `experiment_configuration_sha256 = d5bb759e6deb802e462fe4854f9079ee6cb87a75a75dc4e8510497797ebf6941` |
| generation 16 inference model | `sha256 = 68e7f1e7e8c16e831385ebcef6e6d066bec0d73fbf4bab883cdc02e70bb3f300` (300 full-search visits) |
| generation 30 inference model | `sha256 = c6e83ae2522c38e7c81cf828b8df47b212d4d39db4927c342b9aa39292793075` (400 full-search visits) |
| GPU contention | none; all eight GPUs idle for the whole measurement |
| raw results | `.codex-diagnostics/cut-value-target-study-20260825/raw/*.json` (16 shards) |
| full tables | `generation-16.txt`, `generation-30.txt` in this directory |
| harness | `py/tools/cut_game_value_target_study.py`, `py/tools/cut_game_value_target_report.py` |

No training run was started, stopped or reconfigured. `/workspace/alphazero-engine`,
`/workspace/run-control` and all `training_data` were read only.

## Design

Cut games have no ground truth by construction, so the study manufactures it: play games with the
production self-play recipe but with the ply cap raised from 150 to 400, and for every game that
passes ply 150 record what the target *would* have been against what the game actually did.

Per generation, 11,200 games (4 shards x 2,800 games, 384 concurrent games per shard, one GPU each),
production search parameters resolved from the run's own `resolved-experiment.json`: 300 visits at
generation 16 and 400 at generation 30, Dirichlet 0.25/0.3, exploration constant 1.5, forced
playouts, `retained_root_visit_fraction` 0.6, temperature 1.3 -> 0.1 to ply 60 then greedy,
`full_search_probability` 0.829 (gen 16) / 0.679 (gen 30), fast search forced after ply 200.

For each game reaching ply 150 the harness records, all in the perspective of the side to move at
ply 150:

- `root_value` — the search value of the ply-150 position itself, at the production visit budget;
- `previous_ply_negated_root_value` — the negated root value of the last search production actually
  performs before cutting (ply 149). This is free: it is already in `observation.root_value`;
- `network_root_value` — the raw network evaluation of the ply-150 position, no search;
- `material_scalar` — `approximate_result_score() * current_player`, exactly the production target;
- the true result once the game is played out to natural termination (cap 400).

**Draw handling.** `root_value` is already a scalar: the native search averages the network's
`win - loss` over root visits (`InferenceTypes.hpp:20`, `SearchExecutor.hpp:372`). Draw information
is therefore collapsed before the study sees it, exactly as it is in production. Both candidate
scalars are pushed through the same `WdlTarget.from_scalar` the trainer uses
(`py/src/games/contracts.py:43`, `py/src/training/objective.py:20`) and scored against the one-hot
true result. That makes the Brier and cross-entropy columns a like-for-like comparison of the two
targets as the trainer would actually see them. Sign accuracy is computed on decisive games only;
draws are reported separately as a rate per prediction bucket.

## Result 1: sign accuracy is saturated and therefore tells you nothing

| generation | positions | decisive | majority base rate | root sign acc | material sign acc |
| --- | --- | --- | --- | --- | --- |
| 16 | 2,282 | 1,337 | 0.509 | **0.997** | 0.995 |
| 30 | 1,144 | 969 | 0.523 | **0.989** | 0.981 |

By ply 150 at these generations, whoever is going to win is already obvious to *both* signals. The
material heuristic is sign-correct 99.5% of the time. The question is not sign; it is magnitude and
calibration.

## Result 2: magnitude and calibration — root value wins at both generations

Generation 16, all 2,282 resolved cut positions (win 0.298 / draw 0.414 / loss 0.288):

| predictor | MAE | Brier | cross-entropy | best rescale k | Brier at k |
| --- | --- | --- | --- | --- | --- |
| **root_value at the cut ply** | **0.273** | **0.444** | **0.756** | 0.9 | 0.441 |
| root_value at ply 149, negated (free) | 0.275 | 0.446 | 0.759 | 0.9 | 0.443 |
| raw network value, no search | 0.272 | 0.460 | 0.788 | 0.8 | 0.453 |
| material / 39 (production today) | 0.424 | 0.491 | 0.851 | 1.5 | 0.472 |
| tanh(material_pawns / 12) | 0.329 | 0.537 | 0.906 | 0.7 | 0.502 |
| tanh(material_pawns / 8) | 0.331 | 0.609 | 1.089 | 0.6 | 0.521 |
| tanh(material_pawns / 4) | 0.362 | 0.721 | 1.803 | 0.5 | 0.548 |
| 0.75 root + 0.25 material/39 | 0.311 | 0.444 | 0.766 | 1.0 | 0.444 |
| 0.50 root + 0.50 material/39 | 0.348 | 0.451 | 0.786 | 1.1 | 0.449 |
| constant 0 (uniform WDL) | 0.586 | 0.667 | 1.099 | — | 0.667 |

Generation 30, all 1,144 resolved cut positions (win 0.443 / draw 0.153 / loss 0.404):

| predictor | MAE | Brier | cross-entropy | best rescale k | Brier at k |
| --- | --- | --- | --- | --- | --- |
| **root_value at the cut ply** | 0.235 | **0.193** | **0.374** | 1.6 | 0.177 |
| root_value at ply 149, negated (free) | 0.242 | 0.196 | 0.379 | 1.6 | 0.178 |
| raw network value, no search | 0.238 | 0.205 | 0.386 | 1.6 | 0.193 |
| material / 39 (production today) | 0.580 | 0.388 | 0.707 | 4.6 | 0.211 |
| tanh(material_pawns / 8) | 0.231 | 0.214 | 0.394 | 1.1 | 0.211 |
| tanh(material_pawns / 4) | **0.159** | 0.223 | 0.411 | 0.9 | 0.220 |
| 0.75 root + 0.25 material/39 | 0.321 | 0.219 | 0.441 | 1.9 | 0.178 |
| constant 0 (uniform WDL) | 0.847 | 0.667 | 1.099 | — | 0.667 |

`best rescale k` is the single multiplier on the scalar that minimises Brier — a direct read of how
mis-scaled a predictor is. **k = 0.9 at generation 16 means the root value at generation 16 is
already essentially perfectly calibrated.** That is the answer to the death-spiral concern: the
generation-16 root value is not a noisy quantity that needs damping. The material scalar needs
k = 1.5 at generation 16 and k = 4.6 at generation 30 — it is the mis-calibrated signal, and its
mis-calibration is not even stable across generations.

Calibration table, generation 16, root value (n per bucket, mean prediction, mean realized result):

| bucket | n | mean pred | mean result | W | D | L |
| --- | --- | --- | --- | --- | --- | --- |
| [-1.00, -0.60) | 556 | -0.851 | -0.919 | 0.000 | 0.081 | 0.919 |
| [-0.60, -0.30) | 300 | -0.428 | -0.377 | 0.000 | 0.623 | 0.377 |
| [-0.30, -0.15) | 180 | -0.219 | -0.150 | 0.006 | 0.839 | 0.156 |
| [-0.15, -0.05) | 66 | -0.112 | -0.030 | 0.000 | 0.970 | 0.030 |
| [-0.05, +0.05) | 75 | -0.005 | +0.040 | 0.067 | 0.907 | 0.027 |
| [+0.05, +0.15) | 62 | +0.112 | +0.065 | 0.065 | 0.935 | 0.000 |
| [+0.15, +0.30) | 172 | +0.218 | +0.163 | 0.163 | 0.837 | 0.000 |
| [+0.30, +0.60) | 292 | +0.433 | +0.377 | 0.380 | 0.616 | 0.003 |
| [+0.60, +1.00) | 579 | +0.846 | +0.917 | 0.917 | 0.083 | 0.000 |

Slightly over-confident in the middle, slightly under-confident in the tails; the two cancel, which
is why k lands at 0.9. Same table for material / 39 at generation 16:

| bucket | n | mean pred | mean result | W | D | L |
| --- | --- | --- | --- | --- | --- | --- |
| [-0.60, -0.30) | 495 | -0.423 | **-0.798** | 0.000 | 0.202 | 0.798 |
| [-0.30, -0.15) | 308 | -0.219 | -0.383 | 0.000 | 0.617 | 0.383 |
| [-0.15, -0.05) | 188 | -0.111 | -0.213 | 0.011 | 0.766 | 0.223 |
| [-0.05, +0.05) | 97 | +0.002 | +0.021 | 0.052 | 0.918 | 0.031 |
| [+0.15, +0.30) | 317 | +0.218 | +0.435 | 0.435 | 0.565 | 0.000 |
| [+0.30, +0.60) | 496 | +0.426 | **+0.796** | 0.796 | 0.204 | 0.000 |

Uniformly and substantially under-confident: a two-queen material edge is scored 0.43 when the
realized outcome is 0.80.

## Result 3: the free proxy is as good as a fresh search

`-root_value(ply 149)` — which production already records and stores in the replay as
`observation.root_value` — is statistically indistinguishable from running a fresh search at the
cut position: Brier 0.446 vs 0.444 at generation 16, 0.196 vs 0.193 at generation 30. Bootstrapping
costs no extra search.

The raw network value without any search is only slightly worse (0.460 / 0.205) and still beats
material. The search is not doing the heavy lifting here; the network already knows.

## Result 4: sharpening the material map is the wrong fallback

`tanh(material_pawns / 4)`, the proposed fallback, is **worse than the map it replaces** at
generation 16: Brier 0.721 against 0.491, cross-entropy 1.803 against 0.851. At generation 30 it is
better than today's map (0.223 vs 0.388) and its MAE is the best in the table (0.159). That
inversion is the whole problem: the correct sharpening changes by a factor of about three between
generation 16 and generation 30 (best k 1.5 -> 4.6), because the fraction of ply-150 positions that
still end in a draw collapses from 0.414 to 0.153 as play improves. Any fixed sharpening is
badly wrong at one end of the schedule. A generation-scheduled sharpening would work, but it is a
hand-tuned schedule that the root value gives you for free and correctly.

## Result 5: the "near-uniform targets" premise is not supported, and the real bottleneck is the scalar map

Distribution of the production target's magnitude on cut positions:

| \|material\| pawns | share of cut positions (gen 16) | true draw rate | share (gen 30) | true draw rate |
| --- | --- | --- | --- | --- |
| [0, 1) | 0.025 | 0.947 | 0.035 | 0.700 |
| [1, 3) | 0.038 | 0.828 | 0.084 | 0.604 |
| [3, 6) | 0.141 | 0.766 | 0.205 | 0.222 |
| [6, 10) | 0.223 | 0.638 | 0.195 | 0.157 |
| >= 10 | 0.574 | 0.190 | 0.482 | 0.004 |

Only 6.3% of generation-16 cut positions are within 3 pawns of material equality; 57.4% are ten or
more pawns apart. Mean |material scalar| is 0.324, i.e. a typical production target on a cut game is
about 0.55/0.22/0.22, not 0.333/0.333/0.333. Whatever drove `training/wdl_loss` from 0.476 to 0.786
over generations 15-18, it was not a flood of literally uniform targets. (See the limitations below
before leaning on this: the cut population here is not identical to production's.)

On the small near-balanced stratum, something else shows up. Generation 16, 186 positions within 3
pawns of equality, **85.5% of which are true draws**:

| predictor | Brier | pearson |
| --- | --- | --- |
| root_value | 0.688 | **0.582** |
| material / 39 | 0.679 | 0.311 |
| constant 0 (uniform WDL) | **0.667** | — |

Every scalar predictor loses to a uniform target here, including the good one, even though the root
value clearly carries real signal (pearson 0.58 vs 0.31, sign accuracy 0.889 vs 0.815 on the 27
decisive games). The reason is structural: `WdlTarget.from_scalar` ties the draw mass to
`(1 - |v|) / 3`, so the most draw-confident target it can emit is 0.333/0.333/0.333 while the truth
is 0.08/0.86/0.06. The scalar bottleneck, not the choice of scalar, is what binds in balanced
positions. At generation 30 (57.9% draws in that stratum) root value does win, 0.617 vs 0.660 vs
0.667 for uniform. Emitting a WDL-native root target — the network already produces the triple, the
search collapses it — would recover this. That is a separate change and not required for the
decision below.

## Limitations, stated plainly

1. **Cut rate does not match production.** 0.204 of games reached ply 150 here at generation 16
   (0.102 at generation 30) against 0.547 reported in production at generation 17. Cause: every game
   here starts from the true initial position, while production starts half of its games from
   restart-archive positions carrying an action prefix, which mechanically inflates ply counts. The
   restart archive is per-worker and was not reproducible off the node. Production's cut population
   is therefore probably enriched in games whose start position was filtered to |root_value| <= 0.8,
   which likely enriches exactly the near-balanced stratum of Result 5. This is the main
   external-validity caveat, and it weighs specifically against Result 5's first half — do not treat
   the "premise not supported" line as settled. Results 1-4 compare two predictors on the same
   positions and are robust to the population shift.
2. **Ground truth is the same agent's continuation**, played out under production self-play rules to
   natural termination (cap 400; 6 of 2,288 generation-16 games and 0 of 1,144 generation-30 games
   were still running at 400, reported both ways in the full tables). This is the right counterfactual
   for a value target — it is what the game outcome would have been — but it is not objective truth.
3. **Searches at plies 149 and 150 were forced to be full searches** rather than sampled at
   `full_search_probability`. Mildly favourable to the root-value predictors. Production can match it
   exactly by forcing a full search at the cut ply; or accept the ply-149 proxy, which was measured
   under the same forcing.
4. Resignation is not active at these generations (`first_production_generation: 70`), so no games
   were resigned.
5. Two generations only, 16 and 30, both from run v6. The generation-30 sample (1,144 positions) is
   smaller than generation 16's (2,282) because fewer games reach the cap as play sharpens.

## Recommendation

**Bootstrap.** Replace `adjudicated_wdl`'s material score for `MAXIMUM_PLIES` termination with the
negated root value of the game's last recorded search, and keep everything else. Specifically:

- Use `-observations[-1].root_value` as the terminal scalar for cut games. It costs nothing, it is
  already in the replay record, and it measured within noise of a fresh search at the cut ply.
- Do not damp it and do not blend it at generation 16. Its optimal rescale is k = 0.9; a blend of
  0.75 root + 0.25 material is Brier-neutral at generation 16 (0.444 either way) and strictly worse
  at generation 30 (0.219 vs 0.193). If a safety margin is wanted for the first deployment, 0.75/0.25
  is free early and costs 0.026 Brier at generation 30; anything at or below 0.5 root is a real
  regression.
- The trainer already has the scalar-to-WDL path (`blended_wdl_targets`, `_scalar_to_wdl`) and
  `root_value_blend` already lerps root values into targets from generation 50, so this is a small,
  contained change on the adjudication side rather than new machinery.

**Do not sharpen the material map** as the fallback. `tanh(material_pawns / 4)` is 47% worse than
today's map on Brier at generation 16 and its optimal sharpness moves by a factor of three between
generation 16 and 30.

**On the death-spiral concern specifically:** the measured failure mode is not present. At the exact
generation the run degraded, the root value's sign is right 99.7% of the time and its magnitude is
calibrated to within a 10% rescale. The self-reinforcing risk would show as over-confidence — a best
rescale k well below 1 — and generation 16 shows k = 0.9 while generation 30 shows k = 1.6, i.e. the
network is under-confident later, not over-confident early. The value that would be bootstrapped is
better than the value it replaces, at the generation where it matters.

**Flagged separately, not part of this decision:** `WdlTarget.from_scalar` cannot express a confident
draw, and 86% of near-balanced ply-150 positions at generation 16 are draws. On that stratum no
scalar target beats a uniform one. A WDL-native terminal target is worth a separate look.
