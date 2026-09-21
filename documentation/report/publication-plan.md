# Publication plan

This file is the completion contract for turning the report draft into a final project publication. It complements
the quantitative placeholder chapter; it does not contain provisional result values.

## Claim-to-evidence map

| Publication claim | Required evidence | Allowed wording before closure |
| --- | --- | --- |
| Final model strength | Frozen checkpoint, paired terminal matches, raw games, CI, exact calibration | “Pending terminal evaluation” |
| Improvement over v34 | Same protocol or explicit normalization, both artifact identities | “Final run is active”; no numeric delta |
| Superhuman play | Calibrated fixed-node result with scale caveat and protocol | Use v34 only if clearly labeled predecessor |
| Training cost | Start/stop/downtime audit, node price, exclusions | No live extrapolation |
| Training volume | Reconciled replay/coordinator/trainer counters | No dashboard snapshot |
| INT8 enabled more data | Backend usage, matched backend T results, admitted replay and generation rates | “Designed to improve throughput” |
| Progressive sizing helped efficiency | Stage timing/throughput plus existing small-vs-large T benchmark | Do not claim causal Elo/dollar without counterfactual |
| Replay/restart/auxiliary choices improved strength | Isolated O/S ablation, if any | “Retained in the final bundle” |
| Resignation was safe and useful | Threshold history, continuation outcomes, false-nonloss bound, saved work | “Calibrated with continuation auditing” |
| Reproducible final result | Source/config/archive/checkpoint/backend hashes and evaluation assets | “Recipe available”; result identity pending |

## Required headline tables

1. **Final result summary:** policy-only, 64, 10,000, and chosen high-search condition with opponent, games, W/D/L,
   score, benchmark Elo/CI, latency, parallelism, and artifact ID.
2. **Run identity:** revision, resolved config hash, archive hash, hardware/runtime, dates, effective duration, cost,
   and exclusions.
3. **Training volume:** games, admitted positions, presentations, optimizer steps, effective reuse, final replay
   occupancy, and rejection/quarantine totals.
4. **Stage history:** model stage, visit stage, wall-clock interval, optimizer interval, actor/trainer throughput,
   backend, and promotions/resumes.
5. **Comparison with v34:** only matched rows, with protocol differences shown rather than footnoted away.
6. **Negative-result summary:** technique, tested claim, strongest evidence grade, outcome, and why it was not retained.

## Required figures

The source and derivation of each figure must be machine-readable and retained beside the final archive.

1. Cross-lineage 64-search ladder Elo for v9, v29, v34, the audited successor, and stitched V89–V93.
2. Final-lineage searched and policy-only Elo with uncertainty, promotions, resumes, and backend incidents.
3. Policy, WDL, auxiliary, and total losses against wall clock and optimizer step.
4. Learning rate, gradient norm, and clipping fraction.
5. Games, fresh positions, presentations, generations, and optimizer steps against wall clock.
6. Actor, trainer, materialization, admitted-replay, and generation throughput.
7. Replay occupancy, age percentiles, effective reuse, and surprise/uniform sampling distribution.
8. INT8 legal-policy fidelity and backend usage, annotated with refits, rebuilds, fallbacks, and model transitions.
9. Resignation threshold, triggers, continuation outcomes, and false-nonloss upper bound.
10. Strength versus compute/search depth for the selected checkpoint, with saturated throughput clearly separated
    from interactive latency.

## Final inputs

- fetched and checksum-verified terminal archive;
- exact V89–V93 source/config/resume timeline;
- reconciled run counters and cost accounting;
- selected checkpoint rule and hashes;
- TensorRT/ONNX/template/engine identities and fidelity reports;
- raw terminal match games and result summaries;
- plot extraction scripts or recorded commands;
- final model-card/download identity;
- explicit code and model license decision.

## Writing pass after evidence arrives

1. Populate [Final-run results](07-final-run-results.md) and the project result record first.
2. Write results and discussion without changing historical conclusions to fit the outcome.
3. Update the abstract/conclusion, then replace the root README's temporary v34 showcase with the final result and
   update the documentation index. Retain v34 only as a clearly labeled historical comparison and evidence link.
4. Replace any remaining “current/live/pending” statements with a dated final status.
5. Verify every quantitative claim against the claim map and evidence grade.
6. Pin mutable external documentation and complete bibliography metadata.
7. Run link/anchor validation, render plots, and inspect Markdown/PDF output.
8. Confirm the public model artifact matches the reported checkpoint rather than merely sharing a run name.

## Review gates

- **Scientific:** proxies are not described as Elo; bundled choices are not given isolated causal credit.
- **Statistical:** game counts, pairing, intervals, calibration, and selection rules are explicit.
- **Systems:** rates identify hardware, batch, concurrency, contention, and stage.
- **Reproducibility:** living recipe and frozen result identity are both present and distinguished.
- **Historical:** superseded interpretations remain labeled; negative results and operational failures are not erased.
- **Public language:** benchmark Elo is not presented as FIDE or universal engine rating.
