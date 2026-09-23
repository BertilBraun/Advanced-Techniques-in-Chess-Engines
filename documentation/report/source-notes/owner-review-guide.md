# Owner review guide

You do **not** need to review the full source dossiers. Repository-visible facts, implementation details, benchmark
transcriptions, and source links have already been audited. This guide isolates the decisions that still need project-
owner memory or judgment.

Answer inline with `yes`, `no — ...`, or one or two sentences. If an answer is unknown, say so; the report will retain
the evidence boundary instead of inventing a conclusion.

## Minimum useful pass

For the next review, read only **Priority 1** below: six short questions whose answers cannot be recovered reliably
from repository evidence. That is the only owner-memory pass currently required.

After the remaining result directories are fetched, a second short pass can cover the eight one-paragraph causal
summaries in Priority 2. Priority 3 and the final-result decisions are editorial choices and may wait for the report
outline; they are not prerequisites for continuing the evidence work. You do not need to open any linked dossier
unless a summary looks wrong.

## Priority 1 — facts only the project owner can recover

These are the only questions currently blocking the factual map.

1. **Policy/value trunk sharing.** Do you remember what actually changed in the sharing experiment: separate final
   blocks, fully separate trunks, gradient isolation, or only a proposed design? Any approximate date, branch, config,
   or code phrase would help locate it. Without that, the report will say only that full sharing is implemented and a
   split was proposed. [Context](network-architecture-and-policy.md#policyvalue-trunk-sharing)
2. **Dense policy-head comparison.** Is the missing seven-way result bundle stored outside the repository? If not, may
   we report only the historically recorded qualitative selection of the rank-96, four-channel reduction, without a
   quantitative table? [Context](network-architecture-and-policy.md#dense-reduced-action-policy-heads)
3. **Structured plane policy.** Was the repaired 76-plane head ever trained to convergence or compared online after
   its initialization and ingestion defects were fixed? If not, we will call it *underdetermined/superseded*, not
   rejected. [Context](network-architecture-and-policy.md#structured-76-plane-policy-heads)
4. **Global context.** Was global pooling ever compared directly against squeeze-excitation or no global context in
   chess? If not, it remains a motivated retained component without isolated strength evidence.
   [Context](network-architecture-and-policy.md#global-context-in-convolutional-trunks)
5. **Auxiliary heads.** Were future-search value, irreversible-progress, or legal-move heads used in any completed
   production training whose evidence is not committed? The repository proves that their target layouts exist, but
   not that they contributed to a completed result. [Context](training-data-and-replay.md#auxiliary-target-materialization-and-eligibility)
6. **Historical pretraining.** Was the older self-play pretraining effort important enough to the research conclusions
   to include, or should it remain historical background?

## Priority 2 — consequential interpretations to spot-check

These are not requests to reread the sections. Confirm or correct the summaries below.

1. **Fast/full search:** correct? The Go-inspired scheme was discarded because cheap positions created no direct
   training row; chess supplied terminal outcomes more cheaply and its value objective was already learning well, so
   completing more games did not compensate for the lost policy/value targets. There was no clean chess Elo ablation,
   so this is a mechanism-and-observation conclusion rather than a measured effect size.
   [Context](search-and-inference.md#randomized-fast-and-full-searches)
2. **Graph search:** correct? A complete, corrected implementation was rejected because exact reusable
   transpositions were too rare to repay graph-maintenance overhead. It was not merely proposed, and no strength match
   was run after the corrected economics made continuation unattractive.
   [Context](search-and-inference.md#monte-carlo-graph-search-and-transpositions)
3. **Inference cache:** correct? There were two decisions: a bounded process-local cache was implemented and measured
   unfavorably; a later wider-sharing architecture was declined after an exact-input opportunity audit showed too few
   hits even after reorganizing workers to share more traffic.
   [Context](search-and-inference.md#neural-inference-caching)
4. **Late-game poisoning:** correct? Excluding deep policy rows while assigning one shallow cut value across the game
   over-weighted drawn, unconvertible tails; removing those rows did not instantly undo the learned damage, and searched
   root-value cut targets were the effective repair.
   [Context](evaluation-and-pitfalls.md#transferable-failure-study-late-game-target-poisoning)
5. **Progressive growth:** correct? Training loss was an invalid promotion signal because a candidate receiving more
   presentations could show lower loss while remaining much weaker. Function-preserving growth fixed the initialization
   discontinuity, but the larger model subsequently remained flat; the evidence therefore suggests capacity was not
   the immediate bottleneck, without proving which alternative bottleneck was responsible.
   [Context](training-data-and-replay.md#progressive-candidate-training-and-promotion)
6. **Parallel search:** should the report emphasize this conclusion? Four-way parallelism was the useful practical
   operating point in the measured 1,000-search comparison, while the final strength curve used serial search for the
   shallow points and higher parallelism for the deep points. Consequently, that curve demonstrates attainable
   strength at each budget, not a single fixed-parallelism scaling law.
   [Context](search-and-inference.md#parallel-search-batching-and-tree-retention)
7. **Compilation:** correct? `torch.compile` helped one eager inference diagnostic and attention training, but lost to
   the actual fused TorchScript serving path and hurt convolutional distributed training. The conclusion is boundary-
   specific, not “compilation failed.” [Context](evaluation-and-pitfalls.md#transferable-failure-study-torchcompile-was-not-one-result)
8. **Quantized deployment:** correct? A float-trained model was not safely deployable as INT8; quantization-aware
   training and a quantization-compatible residual design were necessary. Promotion and reporting should use the
   deployed INT8 artifact because the float model overstated strength.
   [Context](network-architecture-and-policy.md#quantization-driven-residual-architecture)

## Priority 3 — choices for the public report

These are editorial decisions, not missing research.

1. **Scope:** should the report be explicitly a chess study conducted on a multi-game platform, with Go appearing
   only as a source of transferred ideas and a small integration check? This is the recommended scope.
2. **Systems depth:** should native ownership, batching, TensorRT, cache work, and runtime alternatives form one major
   systems chapter, while detailed topology sweeps move to an appendix? This would retain the engineering contribution
   without interrupting the learning narrative. [Context](runtime-architecture-alternatives.md#implications-for-report-structure-and-figures)
3. **Failure studies:** which incidents deserve main-text treatment? Recommended: late-game poisoning, invalid
   TensorRT refit, misleading promotion by training loss, and cache measurement correction. Put the seed bug,
   compiler boundary, and replay-materializer wedge in an appendix unless one is especially important to you.
4. **Architecture figure:** should the main SVG show the normal data/control loop plus compact evaluation feedback,
   with recovery and progressive-candidate detail moved to a second figure or appendix? This is the recommended
   legibility tradeoff. [Figure specification](system-architecture-and-figure-dossier.md#proposed-publication-svg)
5. **Terminology:** may the public text use *training quantum* for one optimizer-and-publication unit and reserve
   *checkpoint* for a durable model artifact? This avoids the ambiguous internal term *generation*.

## Final-result decisions now ready for review

The recap closes most former terminal-result placeholders. The following choices should be settled before those
results become headline prose:

1. **Headline calibration:** approve selecting, at each search budget, the Stockfish anchor whose observed score is
   closest to 0.5. This minimizes extrapolation and draw distortion. The two deepest anchors agree within four Elo,
   which is useful calibration evidence—not proof that every shallower anchor is unbiased.
   [Evaluation context](evaluation-and-pitfalls.md#stockfish-calibration-openings-colours-adjudication-and-uncertainty)
2. **Compute-curve wording:** approve reporting 1,658 policy-only Elo and 2,456 / 2,925 / 3,114 / 3,251 Elo across
   four search budgets, with confidence intervals and parallelism stated at every point. Do not describe the curve as
   fixed-parallelism scaling.
3. **Cost denominator:** does the reported $43.20 mean actual billed compute through selection of the reported
   checkpoint, or 60 hours on a stitched axis that excludes discarded work? The report should headline actual spend;
   an effective/stitched time may appear separately but must not be called cost.
4. **Reported model:** approve treating the strongest retained medium-sized checkpoint as the reported model, while
   presenting the later function-preserving larger model as a negative capacity result rather than “the final model.”
5. **Student result:** approve describing the student as 13.4 times smaller and 417 Elo below the teacher at 10,000
   searches after the longer run. Avoid “86% of teacher Elo,” because Elo has no meaningful ratio origin. Tripling
   training improved the matched point estimate by only 14 Elo, inside the overlapping intervals; describe this as
   saturation for this student, replay snapshot, and schedule rather than a universal small-model limit.
6. **Easy-rung explanation:** is the claim that the shallower anchor gap is caused by draws against weak opposition a
   confirmed diagnosis, or only the leading explanation? If it was not separately tested, the report should call it
   an interpretation.

## What you do not need to review

You do not need to check source paths, benchmark transcription, input-plane enumeration, process ownership, replay
schema, configuration constants, or the full evidence ledgers. You also do not need to review final prose yet.

The canonical configuration issue is resolved: its progressive-candidate gate now has the corresponding evaluation
definition and the file loads successfully. Exact historical reproduction still requires the resolved configuration,
source revision, and artifact hashes frozen in the evidence record.

After the questions above are answered, the next useful owner pass is a short narrative outline and selected figures,
not the underlying 5,000 lines of source notes.
