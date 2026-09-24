# Owner review guide

You do **not** need to review the full source dossiers. Repository-visible facts, implementation details, benchmark
transcriptions, and source links have already been audited. This guide isolates the decisions that still need project-
owner memory or judgment.

Answer inline with `yes`, `no — ...`, or one or two sentences. If an answer is unknown, say so; the report will retain
the evidence boundary instead of inventing a conclusion.

## Minimum useful pass

The Priority 1 owner-memory pass, Priority 2 causal review, and Priority 3 editorial scope review are complete. Their
answers and evidence limits are recorded below. The remaining owner questions concern only final-result presentation.
You do not need to open any linked dossier unless a summary looks wrong.

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

## Priority 3 — public-report choices

The editorial scope review is complete.

1. **Scope — chess study:** Chess is the research subject. Small-board Go appears briefly as a platform check, a
   source of transferred ideas, and a failed attempt to use a cheaper game for hyperparameter exploration. The basic
   loop worked, but small-board Go's first-player advantage, shorter games, rapidly learned value target, and different
   tuning needs made it a poor proxy for pushing chess performance. It did not receive its own optimization campaign.
2. **Systems depth — enabling argument only:** Native ownership, batching, C++ MCTS, TensorRT, and the main throughput
   chain remain in one compact chapter because sufficient self-play volume is a precondition for learning. Detailed
   topology sweeps and low-level alternatives move to an appendix/evidence index. The report should not present these
   implementation optimizations as a separate algorithmic breakthrough.
   [Context](runtime-architecture-alternatives.md#implications-for-report-structure-and-figures)
3. **Failure studies — three main incidents:** Keep late-game target poisoning, semantically invalid TensorRT refits,
   and promotion by incomparable training losses. Do not promote the cache-measurement correction, seed bug,
   `torch.compile` boundary comparison, or replay-materializer wedge into publication narratives. Their source notes
   remain as internal evidence where useful.
4. **Architecture figure — normal loop:** The main SVG shows Python/C++ ownership, batched native self-play, replay,
   training, deployment, and compact evaluation feedback. Recovery and progressive-candidate internals are omitted
   from the main figure; recovery does not need a second publication figure.
   [Figure specification](system-architecture-and-figure-dossier.md#proposed-publication-svg)
5. **Terminology — retain generation:** *Generation* is approachable enough for the public narrative and remains the
   term for one training-and-publication cycle. Define it once and reserve *checkpoint* for the durable model artifact
   written at such a boundary. Use *optimizer quantum* only where the distinction is technically necessary.

## Final-result presentation decisions

The owner review of these decisions is complete.

1. **Headline calibration — confirmed:** At each search budget, select the Stockfish anchor whose observed score is
   closest to 0.5. This minimizes extrapolation. The two deepest anchors agree within four Elo, which is useful
   calibration evidence—not proof that every shallower anchor is unbiased.
   [Evaluation context](evaluation-and-pitfalls.md#stockfish-calibration-openings-colours-adjudication-and-uncertainty)
2. **Compute-curve wording — confirmed:** Report 1,658 policy-only Elo and 2,456 / 2,925 / 3,114 / 3,251 Elo across
   four search budgets, with confidence intervals and parallelism stated at every point. Do not describe the curve as
   fixed-parallelism scaling.
3. **Cost denominator — narrow effective cost:** The evidence record and final configuration specify `$0.72/hour`,
   so 60 hours on the accepted-lineage axis equals `$43.20`. This is not actual billed spend: it excludes discarded
   work, experiments, evaluation, distillation, and idle time. Total project experimentation cost was substantially
   larger but will remain unreconciled and unreported. If a billing record later establishes `$0.75/hour`, the
   60-hour product would be `$45.00`; until then the frozen `$0.72` evidence wins.
4. **Reported model — confirmed:** Generation/checkpoint 1026 is the reported teacher. The owner confirms it is
   preserved and published in the project's Hugging Face repository. The larger function-preserving continuation is
   an unresolved capacity experiment, not the reported model.
5. **Student result — confirmed:** Describe the student as 13.4 times smaller and 417 Elo below the teacher at 10,000
   searches after the longer run. Avoid an Elo percentage. Tripling training improved the matched point estimate by
   only 14 Elo inside overlapping intervals; this is saturation for this student, replay snapshot, and schedule.
6. **Opponent-rung disagreement — observed, not diagnosed:** At each model search budget, two Stockfish node limits
   imply somewhat different ratings even though ideal transitive Elo would agree. The discrepancy falls from 128 Elo
   at the shallowest model budget to 4 Elo at the deepest. Draw behavior against the easier opponent is one plausible
   explanation, but it was not isolated. Report the disagreement, select the score nearest 0.5, and do not claim a
   proven draw-bias mechanism.

## What you do not need to review

You do not need to check source paths, benchmark transcription, input-plane enumeration, process ownership, replay
schema, configuration constants, or the full evidence ledgers. You also do not need to review final prose yet.

The canonical configuration issue is resolved: its progressive-candidate gate now has the corresponding evaluation
definition and the file loads successfully. Exact historical reproduction still requires the resolved configuration,
source revision, and artifact hashes frozen in the evidence record.

The next useful owner pass is a short narrative outline and selected figures, not the underlying 5,000 lines of
source notes.
