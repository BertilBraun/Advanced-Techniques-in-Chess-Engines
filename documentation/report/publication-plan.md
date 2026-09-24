# Publication plan

This file is the completion contract for turning the evidence-linked narrative draft into a final publication.
Terminal strength, selected-checkpoint training-volume counters, and the headline figures are complete. Wider
throughput/rejection accounting, some optional figures, and editorial review remain open. Total
project spend is deliberately out of scope; only the narrow accepted-lineage effective cost is reported.

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
| Training cost | Accepted-lineage duration, configured node price, explicit exclusions | `$43.20` is accepted-lineage effective cost, not actual billed project spend; no total-spend claim |
| Training volume | Selected-checkpoint coordinator trajectory and credit ledger; wider node totals separately scoped | Report 3.25 million ingested games, about 209.15 million net materialized positions, and 836.608 million presentations with counter boundaries |
| INT8 enabled more data | Backend usage, matched backend-throughput results, admitted replay and generation rates | “Designed to improve throughput” |
| Progressive sizing helped efficiency | Stage timing/throughput plus existing small-versus-large throughput benchmark | Do not claim causal Elo/dollar without counterfactual |
| Replay/restart/auxiliary choices improved strength | Isolated online-learning or playing-strength ablation, if any | “Retained in the final bundle” |
| Resignation was safe and useful | Threshold history, continuation outcomes, false-nonloss bound, saved work | “Calibrated with continuation auditing” |
| Reproducible final result | Source/config/archive/checkpoint/backend hashes and evaluation assets | Final ONNX, model card, aliases, and checksum index are verified on Hugging Face; the owner confirms the live deployment is current |

## Candidate supporting tables

1. **Final result summary:** policy-only, 64, 10,000, and chosen high-search condition with opponent, games, W/D/L,
   score, benchmark Elo/CI, latency, parallelism, and artifact ID.
2. **Run identity:** revision, resolved config hash, archive hash, hardware/runtime, dates, effective duration, narrow
   effective cost, and exclusions.
3. **Training volume:** games, admitted positions, presentations, optimizer steps, effective reuse, final replay
   occupancy, and rejection/quarantine totals.
4. **Stage history:** model stage, visit stage, wall-clock interval, optimizer interval, actor/trainer throughput,
   backend, and promotions/resumes.
5. **Comparison with the previous four-day baseline:** only matched rows, with protocol differences shown rather than
   footnoted away.
6. **Negative-result summary:** technique, tested claim, strongest evidence dimension, outcome, and why it was not
   retained.

## Visualization strategy

The source and derivation of each figure must be machine-readable and retained beside the final archive.

The report is intended to support a visual-first reading and has no arbitrary figure limit. Three figures are
foundational:

1. normal learning-system loop across Python, C++, TensorRT, replay, training, publication, and evaluation;
2. cross-campaign 64-search ladder Elo, using the existing tracked SVG and keeping internal identifiers in its sidecar;
3. selected-model strength versus search budget, with confidence intervals, both opponent rungs, and parallelism.

Additional main-text candidates include search-allocation alternatives, policy-head representations, replay and
curriculum mechanisms, the end-to-end throughput funnel, late-game poisoning, the quantized-deployment boundary, and
progressive sizing. Include them when they materially clarify the argument rather than to satisfy a count.

Appendix figures may cover raw-versus-stitched lineage provenance, training diagnostics, pipeline/replay health,
quantization fidelity, detailed topology, calibration, and resignation. Omit any panel whose source counters cannot
be reconciled. Dense numerical comparisons should usually be tables, but neither the main text nor appendix has a
fixed table or figure count.

## Final inputs

- fetched and checksum-verified terminal archive;
- exact source/configuration/resume manifest for every accepted and discarded interval;
- reconciled run counters and cost accounting;
- selected checkpoint rule and hashes;
- TensorRT/ONNX/template/engine identities and fidelity reports;
- raw terminal match games and result summaries;
- plot extraction scripts or recorded commands;
- synchronized final model card, aliases, checksum index, and download identity;
- explicit code and model license decision (MIT for original project code, documentation, and final model artifacts).

## Completion pass

1. Finish wider self-play/search-rate, rejection, and stage-time reconciliation beyond the now-tracked
   selected-checkpoint game, replay, trainer-throughput, loss, and learning-rate series; retain explicit gaps where
   records do not permit it. Do not attempt to estimate total project spend from incomplete rental records.
2. Check every quantitative and causal claim against the source artifact and keep proxy, throughput, and Elo results
   distinct. The terminal matrix, headline search curve, and root README result are already integrated.
3. Add figures where they clarify the mechanism, then review visual hierarchy, captions, and accessibility in a
   rendered report. Avoid filling a figure slot merely to reach a count.
4. Pin mutable external documentation and complete bibliography metadata.
5. Validate links and anchors after the topic-first chapter rewrite, then render and inspect a PDF edition if one is
   produced. Markdown remains the editable source.
6. Verify the public Hugging Face model card, aliases, checksum index, and live-site artifact identity. The model
   card now matches the archived 10,000-search parallelism and paired-bootstrap confidence intervals; the final
   ONNX, aliases, checksum index, and MIT metadata are present. The owner confirms the live site's deployed artifact
   is current; this is an owner confirmation, not a second archived hash audit.
7. The owner's MIT decision is recorded in the repository [license](../../LICENSE) and the
   [model-repository license commit](https://huggingface.co/BertilBraun/alphazero-chess/commit/6c068a15bdfaec6b39536e7dc681a55b13f0837e).

## Review gates

- **Scientific:** proxies are not described as Elo; bundled choices are not given isolated causal credit.
- **Statistical:** game counts, pairing, intervals, calibration, and selection rules are explicit.
- **Systems:** rates identify hardware, batch, concurrency, contention, and stage.
- **Reproducibility:** living recipe and frozen result identity are both present and distinguished.
- **Historical:** superseded interpretations remain labeled; negative results and operational failures are not erased.
- **Public language:** benchmark Elo is not presented as FIDE or universal engine rating.
