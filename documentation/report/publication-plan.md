# Publication plan

This file is the completion contract for turning the report draft into a final project publication. It complements
the quantitative placeholder chapter; it does not contain provisional result values.

## Narrative scope

- The publication is a chess study. Small-board Go is limited to transferred ideas, platform validation, and the
  qualitative finding that it was not a useful low-cost proxy for chess hyperparameter optimization.
- Systems engineering is an enabling argument: native MCTS, cross-game inference batching, TensorRT, and replay and
  trainer throughput made sufficient self-play volume possible. Detailed topology sweeps and runtime alternatives
  belong in an appendix or evidence index.
- Main-text failure studies are limited to late-game target poisoning, semantically invalid TensorRT refits, and
  promotion by incomparable training losses.
- The main architecture figure shows the normal Python/C++ data and control loop with compact evaluation feedback.
  Recovery and progressive-candidate internals are not publication figures.
- Use *generation* for a training-and-publication cycle, *checkpoint* for a durable model artifact, and *optimizer
  quantum* only where that lower-level distinction matters.

## Claim-to-evidence map

| Publication claim | Required evidence | Allowed wording before closure |
| --- | --- | --- |
| Final model strength | Frozen checkpoint, paired terminal matches, raw games, CI, exact calibration | Report the completed matrix with its protocol limits |
| Improvement over the previous four-day baseline | Same protocol or explicit normalization, both artifact identities | Compare only matched ladder conditions |
| Superhuman play | Calibrated fixed-node result with scale caveat and protocol | Use “benchmark Elo,” not FIDE or universal engine rating |
| Training cost | Start/stop/downtime audit, node price, exclusions | `$43.20` is accepted-lineage effective cost, not actual billed project spend |
| Training volume | Reconciled replay/coordinator/trainer counters | No dashboard snapshot |
| INT8 enabled more data | Backend usage, matched backend-throughput results, admitted replay and generation rates | “Designed to improve throughput” |
| Progressive sizing helped efficiency | Stage timing/throughput plus existing small-versus-large throughput benchmark | Do not claim causal Elo/dollar without counterfactual |
| Replay/restart/auxiliary choices improved strength | Isolated online-learning or playing-strength ablation, if any | “Retained in the final bundle” |
| Resignation was safe and useful | Threshold history, continuation outcomes, false-nonloss bound, saved work | “Calibrated with continuation auditing” |
| Reproducible final result | Source/config/archive/checkpoint/backend hashes and evaluation assets | Final ONNX identity is verified on Hugging Face; release metadata and aliases remain open |

## Required headline tables

1. **Final result summary:** policy-only, 64, 10,000, and chosen high-search condition with opponent, games, W/D/L,
   score, benchmark Elo/CI, latency, parallelism, and artifact ID.
2. **Run identity:** revision, resolved config hash, archive hash, hardware/runtime, dates, effective duration, cost,
   and exclusions.
3. **Training volume:** games, admitted positions, presentations, optimizer steps, effective reuse, final replay
   occupancy, and rejection/quarantine totals.
4. **Stage history:** model stage, visit stage, wall-clock interval, optimizer interval, actor/trainer throughput,
   backend, and promotions/resumes.
5. **Comparison with the previous four-day baseline:** only matched rows, with protocol differences shown rather than
   footnoted away.
6. **Negative-result summary:** technique, tested claim, strongest evidence dimension, outcome, and why it was not
   retained.

## Figure budget

The source and derivation of each figure must be machine-readable and retained beside the final archive.

Main text contains exactly three figures:

1. normal learning-system loop across Python, C++, TensorRT, replay, training, publication, and evaluation;
2. cross-campaign 64-search ladder Elo, using the existing tracked SVG and keeping internal identifiers in its sidecar;
3. selected-model strength versus search budget, with confidence intervals, both opponent rungs, and parallelism.

The appendix may contain at most four additional figures: raw-versus-stitched lineage provenance, a compact training-
diagnostics panel, a traceable pipeline/replay-health panel, and the quantization-fidelity transition. Omit any panel
whose source counters cannot be reconciled. Use tables for the terminal matrix, rejected-technique catalogue,
parallel-search sweep, student result, configuration, resignation evidence, and detailed topology.

## Final inputs

- fetched and checksum-verified terminal archive;
- exact source/configuration/resume manifest for every accepted and discarded interval;
- reconciled run counters and cost accounting;
- selected checkpoint rule and hashes;
- TensorRT/ONNX/template/engine identities and fidelity reports;
- raw terminal match games and result summaries;
- plot extraction scripts or recorded commands;
- synchronized final model card, aliases, checksum index, and download identity;
- explicit code and model license decision.

## Writing pass after evidence arrives

1. Reconcile the [project result record](../results/final-chess-run.md) with the checksum-covered result artifacts.
2. Write results and discussion without changing historical conclusions to fit the outcome.
3. Update the abstract/conclusion, then replace the root README's temporary predecessor showcase with the final result
   and update the documentation index. Retain the previous four-day baseline only as a clearly labeled historical
   comparison and evidence link.
4. Replace any remaining “current/live/pending” statements with a dated final status.
5. Verify every quantitative claim against the claim map and named evidence dimension.
6. Pin mutable external documentation and complete bibliography metadata.
7. Run link/anchor validation, render plots, and inspect Markdown/PDF output.
8. Refresh the Hugging Face model card, `latest` aliases, and checksum index; the immutable final ONNX already matches
   the reported artifact hash.

## Review gates

- **Scientific:** proxies are not described as Elo; bundled choices are not given isolated causal credit.
- **Statistical:** game counts, pairing, intervals, calibration, and selection rules are explicit.
- **Systems:** rates identify hardware, batch, concurrency, contention, and stage.
- **Reproducibility:** living recipe and frozen result identity are both present and distinguished.
- **Historical:** superseded interpretations remain labeled; negative results and operational failures are not erased.
- **Public language:** benchmark Elo is not presented as FIDE or universal engine rating.
