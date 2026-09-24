# Owner review guide

You do **not** need to review the full source dossiers. Repository-visible facts, implementation details, benchmark
transcriptions, and source links have already been audited. This guide isolates the decisions that still need project-
owner memory or judgment.

Answer inline with `yes`, `no — ...`, or one or two sentences. If an answer is unknown, say so; the report will retain
the evidence boundary instead of inventing a conclusion.

## Minimum useful pass

The **Priority 1** owner-memory pass is complete. The answers and their evidence limits are recorded below. The next
useful review is the eight short causal summaries in Priority 2. Priority 3 and the final-result decisions are
editorial choices and may wait for the report outline. You do not need to open any linked dossier unless a summary
looks wrong.

## Priority 1 — facts only the project owner can recover

These questions are resolved as owner recollection where repository artifacts are unavailable. Such recollections
may explain a decision, but they do not support quantitative claims.

1. **Policy/value trunk sharing:** the project always used a shared trunk and did not run or seriously plan a
   split-trunk treatment. Periods where policy heads dominated the parameter count were head-capacity mistakes, not
   trunk separation. [Context](network-architecture-and-policy.md#policyvalue-trunk-sharing)
2. **Dense policy-head comparison:** the owner remembers roughly ten policy-head comparisons, but does not know where
   the result bundle is. Use the preserved qualitative selection and the controlled comparisons that remain; do not
   reconstruct a numerical table from memory.
   [Context](network-architecture-and-policy.md#dense-reduced-action-policy-heads)
3. **Structured plane policy:** the owner remembers the plane head—recalled as “96-plane,” while the recovered
   implementation uses 76 planes—training in self-play, taking longer, and underperforming. No result artifact is
   currently known. Report this only as a recollected reason for rapid supersession, with the plane-count ambiguity
   visible; retain the preserved supervised measurements as the quantitative evidence.
   [Context](network-architecture-and-policy.md#structured-76-plane-policy-heads)
4. **Global context:** the owner remembers a comparison in which global pooling learned faster, while eventual
   performance was not clearly different. No result artifact is currently known. Treat this as qualitative design
   history, not an isolated Elo effect. [Context](network-architecture-and-policy.md#global-context-in-convolutional-trunks)
5. **Auxiliary heads:** multiple auxiliary heads were used in completed runs, then broadly disabled while debugging a
   period with many interacting failures. This was precautionary de-risking, not an intentional ablation or evidence
   that the heads were harmful. Exact run evidence is not currently known.
   [Context](training-data-and-replay.md#auxiliary-target-materialization-and-eligibility)
6. **Historical pretraining:** exclude it. The reported learning campaign trained from scratch through self-play.
   Rare late-stage parameter ablations resumed a strong checkpoint, but those are diagnostic continuations rather
   than pretraining or part of the main training claim.

## Priority 2 — consequential interpretations

The owner review of these interpretations is complete.

1. **Fast/full search — confirmed:** The Go-inspired scheme was discarded because cheap positions created no direct
   training row; chess supplied terminal outcomes more cheaply and its value objective was already learning well, so
   completing more games did not compensate for the lost policy/value targets. There was no clean chess Elo ablation,
   so this is a mechanism-and-observation conclusion rather than a measured effect size.
   [Context](search-and-inference.md#randomized-fast-and-full-searches)
2. **Graph search — confirmed:** A complete, corrected implementation was rejected because exact reusable
   transpositions were too rare to repay graph-maintenance overhead. It was not merely proposed, and no strength match
   was run after the corrected economics made continuation unattractive.
   [Context](search-and-inference.md#monte-carlo-graph-search-and-transpositions)
3. **Inference cache — confirmed:** There were two decisions: a bounded process-local cache was implemented and measured
   unfavorably; a later wider-sharing architecture was declined after an exact-input opportunity audit showed too few
   hits even after reorganizing workers to share more traffic.
   [Context](search-and-inference.md#neural-inference-caching)
4. **Late-game poisoning — corrected mechanism:** After a configured ply, self-play switched to fast searches whose
   positions were not admitted as proper policy targets. The model therefore saw too few endgames, played those tails
   almost randomly, rarely converted before the early cutoff, and repeatedly fell back to a poor heuristic cut value.
   That value then trained the same weak endgame behavior, forming a non-self-correcting feedback loop. The first
   repair replaced the heuristic with one final full search at the cut position; the later repair removed fast-search
   tails and restored properly searched endgame positions to training.
   [Context](evaluation-and-pitfalls.md#transferable-failure-study-late-game-target-poisoning)
5. **Progressive sizing — stage-dependent conclusion:** The small-to-medium transition has worked well across the
   completed campaigns: the small model buys early throughput, then the medium model supplies capacity when the small
   curve slows. The medium-to-large transition is unresolved. Independently initialized large candidates took too
   long to catch up; function-preserving growth was introduced only in the final investigation and preserved the
   parent's outputs, but the available continuation did not establish that the added capacity generalized better.
   The experiment is insufficient to conclude that capacity was unimportant or that growth initialization is optimal.
   [Context](training-data-and-replay.md#progressive-candidate-training-and-promotion)
6. **Parallel search — report the frontier:** Parallel leaves save wall time but select against stale tree state. The
   relative damage should shrink as the total budget grows because many temporarily suboptimal branches would be
   visited eventually. The project measured about 19 Elo loss for four-way and 45 Elo for sixteen-way parallelism at
   1,000 searches, but it did not measure enough budget/parallelism cells to publish a universal scaling curve. The
   useful future figure is maximum near-free parallelism versus search budget—or the Elo/latency frontier—not a claim
   that the mixed-parallelism terminal curve is fixed-protocol scaling.
   [Context](search-and-inference.md#parallel-search-batching-and-tree-retention)
7. **Compilation — repository-derived:** The owner no longer recalls the details. Preserved benchmarks show that
   `torch.compile` helped one eager inference diagnostic and attention training, but lost to fused TorchScript serving
   and hurt convolutional distributed training. TensorRT later superseded both for production serving. Keep this as a
   bounded historical systems result, not an owner-asserted conclusion or a claim that compilation generally failed.
   [Context](evaluation-and-pitfalls.md#transferable-failure-study-torchcompile-was-not-one-result)
8. **Quantized deployment — corrected rule:** A float-trained model could not simply be converted to faithful INT8;
   the retained INT8 route needed a compatible residual design and QAT. Promotion must measure the artifact intended
   for deployment. If INT8 is the chosen serving path, a stronger float checkpoint cannot stand in for a weaker INT8
   artifact. Conversely, if float deployment is operationally preferable and stronger, the project should deploy and
   report float rather than treating INT8 as intrinsically mandatory.
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

After the remaining causal and editorial choices above are answered, the next useful owner pass is a short narrative
outline and selected figures, not the underlying 5,000 lines of source notes.
