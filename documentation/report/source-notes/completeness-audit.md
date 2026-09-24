# Source-dossier completeness audit

## Verdict

All substantive dossiers are now **ready for project-owner factual review**. The previously missing repository-
visible subjects have been added and checked against their direct sources. The owner is no longer being asked to
rediscover material that the repository already records.

This does not mean the eventual prose is finished or that every retained setting has causal evidence. The owner-memory
questions are resolved, but several answers remain qualitative because their raw artifacts are not known. The final
training recap and checksum-covered tail resolve the checkpoint, progressive-sizing outcome, teacher matrix, and both
student experiments. Remaining work is narrative writing, non-headline figures, accounting, and publication QA.
Check marks in the [topic inventory](topic-inventory.md) record the result of this re-audit.

## Sources checked

This pass compared all files in this directory with:

- the four experiment ledgers and the benchmark-coverage ledger;
- every current benchmark and analysis index;
- the current chess configuration and relevant model, objective, replay, progressive-training, self-play,
  evaluation, and native-search implementation paths;
- the historical research backlog and optimization notes;
- Git history for experiments whose raw result was not preserved in the current benchmark tree; and
- all six preserved tags: the initial release, the pre-runtime-rework snapshot, the completed baseline snapshot, the
  rejected graph-search investigation, the declined inference-cache investigation, and the final adaptive-stopping
  investigation.
- the 2026-09-23 terminal retrospective, deterministic 13-series ladder export, checksum-verified terminal
  TensorBoard archive, and terminal student logs/result files.

Internal tag, run, and checkpoint names are provenance only. They are intentionally not used below as explanatory
labels.

## Dossier readiness by subject

| Dossier | Readiness | Audit result |
| --- | --- | --- |
| Search and inference | Ready for owner factual review | Fixed/mixed budgets, heuristic stopping, predicted allocation, learned stopping, MCTS mechanics, parallelism, graph search, both cache investigations, `torch.compile`, TorchScript, CUDA graphs, TensorRT, and metric boundaries have substantive entries. |
| Network architecture and policy | Owner factual review complete | Input representation and symmetry, all policy families, progressive sizing, function-preserving growth, distillation, trunks, context, value heads, auxiliaries, quantization, and initialization have substantive entries. Missing policy-head/context artifacts remain explicit rather than reconstructed. |
| Training, data, and replay | Ready for owner factual review | Progressive candidate training, the rejected loss gate, match-based promotion, candidate scheduling defects, root-value blending, publication cadence, replay, starts, resignation, cuts, auxiliaries, reanalysis, overlap, storage, and optimizer screens are covered with evidence boundaries. |
| Runtime architecture and current-system figure | Ready for owner factual review | The selected interaction graph is separated from a new investigation dossier covering pipes, queues, clients, fine-grained coroutines, native ownership, direct slots, topology, CUDA graph constraints, and interactive latency. Current progressive semantics were rechecked against code and configuration. |
| Evaluation and pitfalls | Ready for narrative writing | Protocol and failure studies, terminal strength evidence, cross-campaign comparability, and final curves are integrated. Full cost and volume accounting remain open. |

## Repository-visible blockers closed in the follow-up pass

- **Input and augmentation:** the network dossier now defines all 52 input planes, packed binary/scalar storage,
  side-to-move canonicalization, bounded history, repetition, file reflection, castling-plane exchange, checkerboard
  restoration, and policy/legal/auxiliary permutation.
- **Progressive sizing:** the network and training dossiers now distinguish candidate start, extra catch-up work,
  promotion, and manual function-preserving growth. The architecture and system guides agree with the current 50/4
  Elo-per-hour thresholds, six-interval window, two start confirmations, 1.5 candidate multiplier, and the
  match-based promotion gate. They also preserve why unequal candidate training invalidated the former loss gate.
- **Distillation:** raw teacher-output imitation and frozen-replay compression are separate investigations, with
  equal-search, equal-time, and arithmetic equal-compute claims kept distinct. The terminal 20-million-row replay
  student adds a bounded saturation result: tripling training from roughly 7.5 to 23 epochs moved the matched
  10,000-search central estimate only 14 Elo, inside the match uncertainty, while held-out loss had flattened.
- **Value targets:** replay outcome discount, optimizer-time root-value blending, and search-backup discount are now
  separate mechanisms with the target equation and evidence limits recorded.
- **Publication cadence:** the historical 100-step negative decision, retained 500-step boundary, refresh versus
  whole-publication cost, and interaction with replay freshness are now covered without inventing a missing effect
  size.
- **Runtime alternatives:** [`runtime-architecture-alternatives.md`](runtime-architecture-alternatives.md) now gives
  full treatment to the pipe, queue/cache-manager, per-process client, fine-grained coroutine, native ownership,
  direct-slot, actor/process/worker, CUDA-graph, and interactive-service investigations.
- **Service rate versus latency:** saturated self-play/evaluation capacity and single-tree interactive response time
  now have distinct mechanisms, metrics, and preserved measurements.
- **Cross-campaign estimator matching:** the report-source audit now replaces the misleading roughly 102-Elo
  single-rung comparison with a +74.1-Elo three-rung-equivalent plateau comparison and labels the approximately
  ±15-Elo transfer allowance as sensitivity rather than a game-bootstrap interval.
- **Adaptive-rung artifact:** the previous baseline's terminal-looking rise is traced to one threshold-crossing
  outlier followed by a 5,000-to-10,000-node rung transition. Flat policy-only and fixed-dataset controls prevent it
  from being cited as continued learning.
- **Causal allocation:** the matched gap decomposes descriptively into about +30 Elo in the one-expansion policy
  instrument and a further +44 with 64-search tree use. Optimizer evidence remains short, replay-ratio controls remain
  short and unresolved, and the online INT8/FP16 control showed no end-to-end rate win.

## Contradictions and evidence-boundary corrections

### Policy/value trunk sharing was not an experiment

The owner confirms the repository-visible pattern: every model shared the policy/value trunk, and a split trunk was
never a serious project design. The remembered “sharing” issue concerned oversized policy heads—sometimes two
action-sized heads consuming much of a small model—not separate trunks. The report should state shared-trunk design
as an invariant and must not invent a sharing ablation or claim that sharing beat a split alternative.

### Dense-head bake-off has incomplete provenance

Git preserves the seven implemented dense-head variants and the selection commits, but the raw result JSON named by
the selected configuration is absent from the benchmark archive. The recorded rank-96 conclusion is historical
decision evidence, not a publication-grade quantitative result. The policy-head inventory is nevertheless complete
because the missing artifact and its consequence are explicit.

### Plane policy heads have qualitative online history but no recovered clean comparison

The structured mapping was validated, and both gathered and direct-plane forms were implemented. The online failures
also changed initialization, trunk, runtime, ingestion, and action ABI. The owner remembers a later plane head
training through self-play, learning more slowly, and underperforming, but recalls 96 planes where the preserved
implementation uses 76 and does not know the artifact location. The dossier therefore classifies the family as
qualitatively superseded, not quantitatively rejected. Any prose saying the from-to head beat it in a clean online
comparison would exceed the evidence.

### Threshold stopping was audited, not strength-tested

The hand-written stopping rules were reconstructed and rejected on identification and proxy grounds. They were not a
completed production controller defeated in an Elo match. The learned in-search stopper was implemented and tested;
these are distinct investigations and the search dossier now preserves that distinction.

### Cache work consists of two different decisions

A real bounded process-local cache was implemented and measured unfavorably. A later, wider-sharing design was
declined after an unbounded exact-input opportunity audit. Calling the whole subject either “implemented and
rejected” or “only audited” loses one of the two results. The search dossier correctly separates them.

### Graph search was implemented and corrected

The preserved graph-search release contains a full graph implementation, exact-history identity, cycle handling,
rerooting, pruning, correction mechanics, paper/code audit, and low-to-high budget measurements. It must be described
as implemented, corrected, measured, and rejected on reuse economics. It was not merely a cache proposal, and no
post-correction strength match was run.

### Reanalysis and fully asynchronous learning have different statuses

Reanalysis existed in the older replay system but has no controlled efficacy result and was removed with that schema.
It is superseded infrastructure, not a harmful-result finding. Fully asynchronous learning was proposed but not
implemented; the current system overlaps selected self-play actors with synchronized training quanta.

### Auxiliary-head status needs disciplined wording

Next-policy and remaining-length are retained. Future-search-value, irreversible-progress, and legal-move layouts are
implemented in the typed training stack but are not part of the retained recipe and lack independent strength
evidence. The owner confirms that broader auxiliary bundles appeared in completed runs and were removed
precautionarily during a period of multi-cause instability, not by a deliberate ablation. Material, king-safety,
uncertainty, control-map, and similar heads are proposals only. “Implemented target layout,” “trained in a completed
experiment,” and “retained” must not be treated as synonyms.

### Retained configuration is not an ablation table

Global pooling, the exact progressive ladder, restart-state mixture, surprise sampling, random openings, auxiliary
weights, resignation parameters, search constants, and much of the final bundle lack isolated long online Elo
comparisons. The dossiers generally state this correctly. Publication prose must preserve those qualifications and
must not retroactively allocate the terminal run's total gain among individual settings.

The terminal retrospective narrows but does not remove this limitation. High-rate Nesterov SGD is the strongest
single-contributor hypothesis, yet no one-variable optimizer comparison ran longer than 2.5 hours. Replay ratio and
capacity moved together in the completed campaign, scaled post-activation blocks and the policy-loss weight lack
isolated strength tests, and the matched INT8/FP16 online pair does not explain the campaign's TensorRT-over-
TorchScript throughput difference. The +30/+44 policy/search decomposition is an outcome decomposition, not a
feature attribution.

### Progressive-sizing conclusion is stage-dependent

The small-to-medium transition worked repeatedly and is the well-supported progressive-sizing result. The
medium-to-large transition remains unresolved. An independently initialized candidate was promoted by an invalid
loss comparison and remained far weaker in play. Function-preserving growth removed the initial relearning deficit,
restored INT8 fidelity through QAT, and reached parity, but the limited continuation then stayed flat. The reported
checkpoint remains the 14-by-160 model; the evidence cannot distinguish insufficient training, post-growth
optimization, target or replay limits, or lack of useful additional capacity.

### Standalone promotion configuration repair is complete

The final YAML now includes the `progressive_candidate` evaluation referenced by its promotion gate and loads as a
self-contained configuration. Exact historical reproduction still requires the frozen resolved configuration and
source revision rather than whichever future revision the living file reaches.

## Resolved scope choices found outside the original inventory

The audit added explicit inventory rows for augmentation/canonicalization and model-publication cadence, and both are
now covered. The project owner resolved the remaining scope choices as follows:

- **Go platform work.** The report is a chess study. Small-board Go appears only as a platform check, a source of
  transferred ideas, and a qualitative example of failed hyperparameter transfer; it is not a second research result.
- **Historical pretraining.** Exclude it from the report. The claimed campaign trained from scratch through self-play.
  Rare checkpoint-resumed late-stage parameter ablations are diagnostic continuations, not pretraining.
- **Systems work.** Model refresh, batching, native ownership, TensorRT, and replay/trainer throughput appear only as
  enabling infrastructure. Detailed topology and runtime alternatives move to supporting material.
- **Mixed-precision training.** BF16 is part of the retained trainer and inference recipe, but the dossier has no
  self-contained precision investigation. Add it to the systems section if the historical tests support a decision;
  otherwise describe it only as configured method.

## Owner-memory questions resolved

The owner confirms that trunks were always shared; broader policy-head comparisons, a slower underperforming plane
head, a faster-learning global-pooling comparison, and completed auxiliary-head runs existed, although several raw
result bundles are not currently known. Those memories are recorded as qualitative decision history rather than
quantitative evidence. Historical pretraining is out of report scope. Causal and editorial scope review is complete;
only the remaining final-result presentation choices require owner judgment.

## Current publication boundary

The terminal teacher and student matrices, selected checkpoint and artifact identity, cost basis, matched ladder
comparison, and selected-checkpoint training trajectory are now integrated into the report, result record, and root
README. Their raw or derived evidence is indexed in the
[final-run evidence directory](../../evidence/final-chess-20260923/README.md). The cross-campaign ladder figure trims
the final recipe at 2.5 days and the previous four-day baseline at 3.0 days while retaining the untrimmed export as
provenance. The abstract, conclusion, and headline README claims are no longer waiting for terminal evaluation.

Publication work that remains is narrower: reconcile wider actor/search and rejection telemetry where the local
archive permits it; settle the operational timing definition for the parallel-search speedup; finish optional
figures and editorial review; verify release metadata, the public model card and live artifact; and record the
owner's code/model licensing decision. The [publication plan](../publication-plan.md) tracks these gates. Missing
causal ablations must remain limitations, not be inferred from the bundled final result.
