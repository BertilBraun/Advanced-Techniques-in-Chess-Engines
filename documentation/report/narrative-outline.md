# Technical report narrative outline

This is the short owner-review artifact for the final report. It defines the story, claims, tables, and figure budget
before the evidence-bearing drafts are rewritten into publication prose. It intentionally omits internal run labels,
implementation archaeology, and benchmark transcription.

## Central question and answer

**Question:** How strong can an AlphaZero-style chess system become when self-play, training, and evaluation share
eight consumer GPUs?

**Answer:** Training entirely from self-play produced a 6.3-million-parameter chess model that measured 1,658
benchmark Elo without search and 2,456, 2,925, 3,114, and 3,251 Elo across four increasing search budgets. Under a
matched ladder estimator, the completed recipe improved approximately 74 Elo over the previous four-day baseline.
A 470,295-parameter distilled student retained 2,697 benchmark Elo at 10,000 searches. The result did not come from
one isolated trick: search, training targets, replay, model design, and throughput had to work as one learning system.

The ratings are protocol-specific fixed-node Stockfish calibrations, not FIDE ratings or universal engine-list Elo.
The `$43.20` figure is the accepted-lineage cost of 60 effective hours at the verified `$0.72/hour` rate, not total
project expenditure.

## 1. Why this study exists

Introduce AlphaZero as a closed learning loop whose useful rate is determined by fresh searched positions,
optimization, and sufficiently sensitive evaluation—not merely neural-network throughput. State the compute
constraint and the decision criterion: playing strength per wall-clock hour on one shared eight-GPU node.

Chess is the study. Small-board Go receives one paragraph: it validated the shared platform and supplied ideas such
as randomized fast/full searches, but it was not a useful cheap proxy for chess hyperparameter optimization. Its
first-player advantage, shorter games, rapidly learned value target, and game-specific tuning needs made a separate
optimization programme necessary, so the project retained its focus on chess.

The introduction should claim:

- a complete from-scratch self-play result;
- a measured set of retained and rejected search, model, and data techniques;
- an end-to-end throughput design that made the learning experiment feasible;
- a frozen terminal evaluation and reproducible selected artifact.

It should not claim a new general-purpose search algorithm, universally optimal hyperparameters, or comparability
with unrestricted contemporary engines.

## 2. One learning loop under a compute constraint

Orient the reader with the normal system loop:

```text
published model → batched native self-play → replay → distributed training
                → deployment export → paired evaluation → model selection
```

C++ owns chess state, legal moves, MCTS trees, and the latency-sensitive batched search loop. Python owns
configuration, replay, training, publication, and evaluation. TensorRT evaluates batched leaves. Replay controls the
freshness, reuse, and diversity of the training distribution. Training and self-play share the same accelerators, so
a saved search is useful only when it reaches the wall-clock learning bottleneck.

Define *generation* as one training-and-publication cycle and *checkpoint* as its durable model artifact. Keep the
evaluation-method box short: paired colour-reversed openings, raw W/D/L, confidence intervals, fixed-node Stockfish
anchors, and a strict distinction between proxy, throughput, target fidelity, and playing-strength evidence.

## 3. Spending search where it matters

Tell this chapter through hypotheses rather than chronology:

1. **Fixed budgets** establish the strength/compute curve and remain the reliable baseline.
2. **KataGo-style fast/full searches** transferred poorly. Cheap positions were not training targets, while chess
   already supplied shorter games and tractable terminal value supervision; the scheme discarded useful rows without
   compensating value-target benefit.
3. **Rule-based and predicted adaptive budgets** either lacked an identifiable safe signal or improved a proxy
   without improving learning.
4. **Learned stopping** saved search but did not shorten the overlapped training critical path enough to matter.
5. **Parallel leaves** trade strength for latency; the acceptable count depends on the total search budget.
6. **Monte Carlo graph search** was implemented but exact transposition reuse was too sparse to repay its overhead.
7. **Inference caching** was rejected after both an implemented local cache and a wider opportunity audit found too
   little exact-input reuse.

End with the retained fixed-budget search design and state that all negative results are workload-specific rather
than universal rejections of the techniques.

## 4. Choosing what the network predicts

Use policy representation as the chapter's spine. Start from the 1,880-action chess output problem, then explain the
genuinely different head families: large dense reduced-action projections, structured plane heads, and the retained
from-to attention policy. The from-to head has the strongest preserved comparison; the structured head is remembered
as slower and weaker in self-play, but missing artifacts prevent a reconstructed numerical result.

Then cover the shared convolutional trunk, global context, WDL/value prediction, training-only auxiliaries, and the
CNN-versus-attention workload comparison. Trunk sharing is an implemented invariant, not an ablated result. Global
pooling appears to have learned faster, but no isolated final-strength effect is preserved. Auxiliary heads were
removed precautionarily during a difficult debugging period rather than shown to be harmful.

Quantization belongs here as an architectural constraint, not merely an export step: activation caps, scaled
post-activation residual blocks, and QAT made faithful INT8 deployment possible. Finish with progressive sizing:
small-to-medium progression worked reliably, while medium-to-large growth remains unresolved. Function-preserving
growth removed the initial relearning discontinuity but did not establish better generalization in the limited
continuation.

## 5. Getting more learning from each game

Organize replay and curriculum around information efficiency:

- capacity growth, freshness, reuse, and publication cadence;
- row admission versus sampling probability versus loss weighting;
- shallow random openings and difficult restart positions;
- policy-surprise sampling and reservation of consequential states;
- resignation with continuation auditing;
- game caps, cut targets, and target eligibility;
- remaining-length and next-policy auxiliary targets;
- why reanalysis and a fully asynchronous learner were not retained.

The main conclusion is that more presentations are not necessarily more information. Restart states and replay
sampling target difficulty through different mechanisms. Most retained curriculum choices belong to one successful
bundle and should not receive invented isolated Elo credits.

## 6. Making the loop fast enough

Keep this chapter deliberately short. Native C++ search removed Python from the leaf-level hot path. Hundreds of
interleaved games supplied large GPU batches. TensorRT provided the final serving compiler, while quantization-aware
training preserved the policy closely enough for deployment. Columnar replay and persistent distributed training
kept the learner supplied with data.

Follow throughput through the entire chain—searches, games, admitted positions, optimizer cadence, and Elo/hour.
Avoid presenting kernel speed as learning speed. Detailed worker-count sweeps, compiler alternatives, recovery,
queue mechanics, and topology history belong in supporting material.

## 7. Three failures that changed the method

This is the only intentionally temporal chapter because diagnosis requires sequence.

### Late-game target poisoning

Fast-search-only tails stopped contributing properly searched endgame positions. Weak late-game play then failed to
finish games, an early cutoff supplied a poor heuristic value, and that target trained the same weak behavior. First,
one final full search replaced the heuristic at the cut position. Later, the fast-search tail was removed so searched
endgame positions returned to replay. The transferable lesson is that search eligibility, row admission, and terminal
value provenance form one feedback loop.

### Mechanically valid but semantically invalid TensorRT refits

TensorRT accepted a refit whose changed quantization scales invalidated assumptions optimized into the template. API
success and complete weight accounting did not imply a correct chess model. Real-position legal-policy KL, top-one
agreement, and WDL checks therefore became publication gates. The transferable lesson is to validate deployment
artifacts as semantic models.

### Promotion using incomparable training losses

A larger candidate received extra presentations of the same replay data, so its lower training loss was structurally
advantaged and did not imply equal playing strength. Loss-based promotion admitted a much weaker model. Head-to-head
matches between the artifacts intended for deployment replaced loss parity. Function-preserving growth separately
addressed the cost of relearning the parent's behavior. The transferable lesson is that unequal optimization exposure
invalidates loss as a promotion criterion.

## 8. The integrated chess recipe

Present the final system as one compact table with six rows:

| Area | Retained mechanism | Evidence language |
| --- | --- | --- |
| Representation and model | 52-plane input, shared scaled-post-activation CNN, global context, from-to policy, WDL value | Selected architecture; component evidence varies |
| Search and self-play | Fixed visit schedule, PUCT, exploration noise, forced playouts, retained tree, bounded parallel leaves | Retained after search investigations |
| Data curriculum | Random openings, difficult restart states, resignation auditing, searched cut values | Retained bundle; mostly not isolated |
| Replay and training | Growing columnar replay, uniform/surprise mixture, Nesterov SGD, bfloat16, QAT | Mechanism and assembled-run evidence |
| Progressive sizing | Fast small model followed by medium model; match-based promotion | Small-to-medium supported; large-model benefit unresolved |
| Deployment and evaluation | TensorRT INT8 searched play, float policy-only export, paired Stockfish brackets | Artifact-specific terminal evidence |

The canonical reproduction entry point is the living final chess configuration. Exact result reproduction uses the
frozen source, configuration, checkpoint, and evaluation artifacts instead.

## 9. What the completed system achieved

Lead with the matched-estimator improvement and the cross-campaign curve, then present the selected checkpoint and
terminal search curve. Discuss diminishing returns across search decades and label the parallel-search count at every
point. The two opponent rungs disagree at shallow budgets; report the discrepancy without assigning a cause and use
the rung closest to a 50% score.

Present the distilled student as a separate compression result: 13.4 times fewer parameters, 417 Elo below the
teacher at the matched 10,000-search point, and only 14 Elo improvement after tripling training within overlapping
intervals. Do not express Elo as a percentage.

Close the results with the capacity boundary: the selected checkpoint is the medium model. The larger continuation
reached parity but did not establish a gain in the available training window; it does not prove that capacity was
irrelevant. Separate the `$43.20` accepted-lineage cost from unknown total project expenditure.

## 10. What remains uncertain

Combine limitations, reproducibility, and conclusion:

- benchmark Elo depends on the fixed-node calibration and match protocol;
- many retained choices lack isolated full-run ablations;
- search and systems conclusions depend on this model, hardware, and concurrency regime;
- accepted-lineage time and cost are not total project compute;
- large-model optimization and the precise safe-parallelism frontier remain open;
- public artifacts require source, configuration, model, inference, and evaluation identities rather than mutable
  aliases.

End with three takeaways:

1. Information quality mattered more than raw replay volume.
2. Search-saving proxies mattered only when their savings reached wall-clock learning.
3. Architecture, training, deployment, and evaluation had to be designed together.

## Publication tables

Keep the main text to four tables:

1. selected checkpoint and terminal strength summary;
2. integrated retained recipe;
3. substantial rejected or inconclusive techniques;
4. distilled-student result.

The complete terminal matrix, exact configuration, topology sweeps, and evidence manifests remain linked supporting
tables rather than duplicated in the narrative.

## Figure budget

Use three main-text figures:

1. **Learning-system loop — new SVG.** Python orchestration, native C++ self-play/MCTS, TensorRT leaf inference,
   replay, distributed training, publication, and compact evaluation feedback.
2. **Cross-campaign 64-search progression — ready.** Reuse
   [`chess-ladder-progress.svg`](../showcase/chess-ladder-progress.svg). Its curves are descriptive; the approximately
   74-Elo comparison comes from the matched-estimator audit rather than subtracting displayed endpoints.
3. **Selected-model strength versus search budget — new.** Logarithmic search axis, Elo confidence intervals, both
   opponent-rung estimates shown lightly, and parallelism labeled at every searched point.

Allow at most four appendix figures:

- raw versus stitched lineage with excluded intervals;
- training diagnostics: losses, learning rate, gradient norm, and clipping;
- pipeline/replay health where counters can be reconciled;
- quantization-fidelity transition supporting the TensorRT failure study.

Use tables—not figures—for the complete terminal matrix, negative-result catalogue, parallel-search sweep, student
results, detailed topology, configuration, and resignation evidence. The older showcase figures are predecessor-era
artifacts and should not enter the report unchanged.

## Minimum owner review before prose expansion

Review only:

1. the opening question, answer, and contribution claims;
2. the one-paragraph Go scope;
3. the concluding decision paragraph of the search, model, and data chapters;
4. the three failure-study causal sequences;
5. the matched-estimator improvement and cost wording;
6. the plateau/capacity interpretation;
7. the three concluding takeaways;
8. the three-figure budget.

Benchmark transcription, implementation details, evidence links, manifests, and appendix material do not require
owner line review.
