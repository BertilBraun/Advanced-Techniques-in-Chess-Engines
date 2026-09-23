# Source-dossier completeness audit

## Verdict

All substantive dossiers are now **ready for project-owner factual review**. The previously missing repository-
visible subjects have been added and checked against their direct sources. The owner is no longer being asked to
rediscover material that the repository already records.

This does not mean the eventual prose is finished or that every retained setting has causal evidence. One disputed
experiment still requires project-owner memory or a recovered artifact, and several historical conclusions retain
explicit provenance limits. The final training recap now resolves the reported checkpoint and the progressive-sizing
outcome; the teacher evaluation table is integrated, while several post-pull result directories, the final figures,
and full cost accounting remain open. Check marks in the [topic inventory](topic-inventory.md) record the result of
this re-audit.

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

Internal tag, run, and checkpoint names are provenance only. They are intentionally not used below as explanatory
labels.

## Dossier readiness by subject

| Dossier | Readiness | Audit result |
| --- | --- | --- |
| Search and inference | Ready for owner factual review | Fixed/mixed budgets, heuristic stopping, predicted allocation, learned stopping, MCTS mechanics, parallelism, graph search, both cache investigations, `torch.compile`, TorchScript, CUDA graphs, TensorRT, and metric boundaries have substantive entries. |
| Network architecture and policy | Ready for owner factual review | Input representation and symmetry, all policy families, progressive sizing, function-preserving growth, distillation, trunks, context, value heads, auxiliaries, quantization, and initialization have substantive entries. The remembered split-trunk experiment remains an explicit owner-memory question rather than an invented result. |
| Training, data, and replay | Ready for owner factual review | Progressive candidate training, the rejected loss gate, match-based promotion, candidate scheduling defects, root-value blending, publication cadence, replay, starts, resignation, cuts, auxiliaries, reanalysis, overlap, storage, and optimizer screens are covered with evidence boundaries. |
| Runtime architecture and current-system figure | Ready for owner factual review | The selected interaction graph is separated from a new investigation dossier covering pipes, queues, clients, fine-grained coroutines, native ownership, direct slots, topology, CUDA graph constraints, and interactive latency. Current progressive semantics were rechecked against code and configuration. |
| Evaluation and pitfalls | Ready for current owner factual review | Protocol and failure studies are strong. The explicitly marked terminal-results fields, cross-campaign comparability audit, final curves, cost, and reproducibility identity must wait for the preserved final archive. |

## Repository-visible blockers closed in the follow-up pass

- **Input and augmentation:** the network dossier now defines all 52 input planes, packed binary/scalar storage,
  side-to-move canonicalization, bounded history, repetition, file reflection, castling-plane exchange, checkerboard
  restoration, and policy/legal/auxiliary permutation.
- **Progressive sizing:** the network and training dossiers now distinguish candidate start, extra catch-up work,
  promotion, and manual function-preserving growth. The architecture and system guides agree with the current 50/4
  Elo-per-hour thresholds, six-interval window, two start confirmations, 1.5 candidate multiplier, and the
  match-based promotion gate. They also preserve why unequal candidate training invalidated the former loss gate.
- **Distillation:** raw teacher-output imitation and frozen-replay compression are separate investigations, with
  equal-search, equal-time, and arithmetic equal-compute claims kept distinct.
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

## Contradictions and evidence-boundary corrections

### Policy/value trunk sharing requires owner evidence

The project owner recalls an experiment on transformer policy/value trunk sharing. Current code, configuration,
benchmark records, experiment ledgers, and searched Git history show fully shared trunks and proposals for partial
separation, but no implemented split-trunk module or controlled sharing comparison. The dossier currently says so.
This is a direct conflict between repository evidence and project memory, not a reason to erase the recollection.

Required resolution: obtain a branch, commit, deleted artifact, configuration, or a precise description of what was
changed. Until then, the report may say only that full sharing is the implemented baseline and splitting was proposed;
it must not say either that a split-trunk experiment failed or that sharing won.

### Dense-head bake-off has incomplete provenance

Git preserves the seven implemented dense-head variants and the selection commits, but the raw result JSON named by
the selected configuration is absent from the benchmark archive. The recorded rank-96 conclusion is historical
decision evidence, not a publication-grade quantitative result. The policy-head inventory is nevertheless complete
because the missing artifact and its consequence are explicit.

### Plane policy heads were not cleanly rejected

The structured mapping was validated, and both gathered and direct-plane forms were implemented. The online failures
also changed initialization, trunk, runtime, ingestion, and action ABI. The dossier correctly classifies the repaired
plane family as underdetermined/superseded. Any later prose saying the from-to head beat the plane head in a clean
online comparison would exceed the evidence.

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
evidence. Material, king-safety, uncertainty, control-map, and similar heads are proposals only. “Implemented target
layout,” “trained in a completed experiment,” and “retained” must not be treated as synonyms.

### Retained configuration is not an ablation table

Global pooling, the exact progressive ladder, restart-state mixture, surprise sampling, random openings, auxiliary
weights, resignation parameters, search constants, and much of the final bundle lack isolated long online Elo
comparisons. The dossiers generally state this correctly. Publication prose must preserve those qualifications and
must not retroactively allocate the terminal run's total gain among individual settings.

### Capacity conclusion is bounded

The independently initialized larger candidate was promoted by an invalid loss comparison and remained far weaker
in play. Function-preserving growth removed the relearning deficit, restored INT8 fidelity through QAT, and reached
parity, but the larger continuation then stayed flat. The reported checkpoint remains the 14-by-160 model. This is
evidence that capacity was not binding under the tested learning-rate floor, search targets, and replay stream; it
is not evidence that larger networks are generally ineffective.

### Standalone promotion configuration needs one repair

The current standalone final YAML configures a match gate with definition ID `progressive-candidate`, but its
evaluation-definition list contains no `progressive_candidate` entry. The campaign continuation configuration does
contain that paired-match definition. Because configuration loading does not cross-check the reference, the
standalone YAML loads but could never accumulate promotion evidence. Correct the final YAML before presenting it as
an executable reproduction recipe.

## Nonblocking scope choices found outside the original inventory

The audit added explicit inventory rows for augmentation/canonicalization and model-publication cadence, and both are
now covered. The following are project-owner scope choices rather than missing repository research:

- **Go platform work.** A 7x7 baseline and KataGo evaluation integration exist. Decide whether the report is about
  the chess campaign on a multi-game platform or about the complete repository. If chess-focused, state the scope and
  use Go only where it supplies a transfer comparison.
- **Historical pretraining.** An older self-play pretraining path and result note exist. Determine whether it belongs
  to the final research story or only historical background; it has not been synthesized in a dossier.
- **Model refresh and interactive serving.** The runtime dossier now covers them. The owner can still choose whether
  they appear as enabling infrastructure or as full systems results in the publication.
- **Mixed-precision training.** BF16 is part of the retained trainer and inference recipe, but the dossier has no
  self-contained precision investigation. Add it to the systems section if the historical tests support a decision;
  otherwise describe it only as configured method.

## Questions that genuinely require project-owner memory

These are the remaining questions the repository audit cannot answer safely:

1. What exactly was changed in the recalled transformer policy/value trunk-sharing experiment? Which branch,
   checkpoint, or approximate date could locate it, and did it split the trunk or merely compare head gradients or
   capacity?
2. Is there an uncommitted or external result bundle for the seven-way dense policy-head bake-off?
3. Was the repaired 76-plane policy head ever trained to convergence or compared online after its initialization and
   ingestion defects were fixed?
4. Was global-pooling context ever compared directly with squeeze-excitation or no context in chess, outside the
   bundled architecture changes?
5. Should historical self-play pretraining and the paused Go work be part of the paper's claimed experimental scope?
6. Were any of the future-search-value, irreversible-progress, or legal-move auxiliary heads used in a completed
   production run whose evidence is not committed?

These questions are suitable for owner review. No known repository-visible subject remains for the owner to
rediscover before that review.

## Remaining result-integration work

The final recap has supplied the terminal checkpoint choice, cost basis, principal strength matrix, capacity outcome,
and archived artifact locations. The following work remains before publication:

- reconcile the final policy-only, fixed-search, moderate/deep-search, and very-deep-search results with their raw
  result files and uncertainty calculations;
- freeze the exact serving-artifact identities and re-fetch evaluation outputs produced after the evidence pull;
- extract complete optimizer volume, admitted data, effective replay reuse, and systems telemetry from the archive;
- loss, learning-rate, Elo, replay-age, resignation, cut, throughput, and resource curves;
- the remaining shared-protocol audit; and
- render the largest model's TensorRT/QAT recovery and fidelity history without implying it is the reported model.

These pending results do not block factual review of any source dossier. They do block final evaluation, abstract,
headline README claims, and conclusion prose.

The cross-campaign ladder figure is complete. Its deterministic input trims the final recipe at exactly 2.5 days
and the previous four-day baseline at exactly 3.0 days while preserving the untrimmed export as provenance.
