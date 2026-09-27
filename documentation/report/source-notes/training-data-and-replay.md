# Training, data generation, and replay source dossier

This is a source dossier, not report prose. It is organized by technical question rather than by run chronology.
Internal run identifiers are omitted from the explanation. Some source paths contain historical identifiers because
that is how the preserved artifact is named; those identifiers are provenance, not concepts a reader must learn.

Four operations that are easy to conflate are kept separate throughout:

1. **Generation** decides which positions are searched and what observations a completed game contains.
2. **Admission** decides which observations become durable replay rows and which targets on those rows are eligible.
3. **Selection** decides which durable rows are drawn for an optimizer batch.
4. **Weighting** changes how strongly a selected row contributes to the objective.

A fifth quantity, **presentation credit**, controls when optimization is allowed. It accounts for training work but
does not change the stored rows or their sampling probabilities.

## Optimizer family, learning-rate schedule, warmup, clipping, and weight decay

### Question

- Which optimizer and schedule learn the changing self-play distribution reliably, and which conclusions can be
  drawn from short stationary-replay screens rather than online strength?

### Alternatives and mechanisms

- Successful earlier training used AdamW. Controlled stationary-replay work compared AdamW against SGD with
  momentum and then screened several Nesterov-SGD schedules.
- The retained optimizer is SGD with momentum `0.9`, Nesterov enabled, and weight decay `1e-4`. The global batch is
  2,048 positions across eight data-parallel ranks.
- The retained base learning rate warms linearly from zero to the scheduled rate over the first 1,000 optimizer
  steps. The scheduled rate itself decays linearly from `0.1` to `0.01` across the first 500,000 optimizer steps and
  remains at `0.01` afterward.
- Gradient clipping uses a global norm ceiling of `1.0`. Clipping is therefore part of the optimizer actually used,
  not merely a diagnostic.
- Quantization folding prompted separate investigations of phase-specific warmup. Historical experiments folded
  BatchNorm early, reset the optimizer, and warmed a post-fold rate from a small floor. The retained deployment-copy
  design no longer changes the trainable model at the fold boundary, so those post-fold screens explain a failure
  mode and design transition rather than the active training schedule.
- A much older implementation documented a one-cycle policy. It is historical implementation evidence, not evidence
  for the current recipe and not a controlled comparison against the retained linear schedule.

### Evidence and results

- In a 5,000-step frozen-replay comparison using the same sampled indices, AdamW at `0.002` reached total loss
  `2.73843`; Nesterov SGD at `0.1` with a norm ceiling of `5.0` reached `2.88698`. Relaxing an inherited `0.5` clip to
  `5.0` improved SGD by `0.03683`, but it remained `0.14855` behind AdamW. This establishes faster early fitting for
  AdamW in that stationary proxy, not better online generalization.
- A later 12,000-step Nesterov screen held initialization and sample order fixed. Warming the post-fold rate to
  `0.06` produced total loss `2.72291`, compared with `2.79997` for an immediate `0.02`. The higher-rate arm also
  escaped persistent clipping more quickly. All arms suffered a fold-transition gradient shock.
- A four-arm pre-fold factorial found that delaying a destructive fold from step 1,000 to step 3,000 improved total
  loss from `2.82023` to `2.77468` under the otherwise matching historical schedule. Every arm continued learning,
  and every first post-fold step was clipped.
- A post-fold sweep ranked target rates monotonically over the measured horizon: total loss improved from `2.76752`
  at `0.02` to `2.72231` at `0.08`. None diverged. Again, this was a stationary-replay proxy, not chess strength.
- An earlier fixed-teacher study favored AdamW with cosine decay over both flatter AdamW schedules and an untuned SGD
  schedule. The SGD arm was additionally confounded by an architecture transition and an untuned peak, so it cannot
  support a general rejection of SGD.
- The terminal retrospective supplies short online corroboration for the retained high-rate Nesterov choice, but not
  a long causal estimate. A roughly one-hour high-rate Nesterov arm was about 95 Elo ahead of its AdamW control; two
  lower-rate Nesterov arms were 137 and 198 Elo behind the high-rate arm at 40 and 60 minutes. A later campaign using
  the predecessor recipe was about 150 Elo ahead of the mature baseline after 8–12 hours under the same single-rung
  protocol. These comparisons make learning-rate magnitude and optimizer family plausible contributors, but every
  direct arm was short and the longer comparison changed a bundle rather than one variable.
- The final model reached the predecessor's terminal policy/WDL training losses in about 40% as many generations.
  That is consistent with faster fitting under the changed training recipe. It is not an optimizer-only result:
  replay reuse, replay capacity, policy-loss weight, quantization-aware training, and publication cadence also
  differed, while generation count itself changes meaning when reuse changes.

### Decision rationale

- Nesterov SGD was retained as a long-horizon generalization hypothesis after proving that it could fit the target
  stably. The project does not have a matched long online AdamW-versus-SGD result that assigns causal Elo credit.
- The long pre-fold phase and deployment-copy folding avoid the optimizer discontinuity revealed by the short fold
  screens. Their winning numerical schedules are not copied literally into the retained recipe.
- Warmup, clipping, optimizer, batch size, replay reuse, and data freshness interact. A loss-per-step result is not a
  wall-clock learning result when a different optimizer changes trainer time or when reuse changes the fresh-data
  stream per update.

### Pitfalls and unknowns

- The strongest frozen-replay settings were selected on one or two seeds and a stationary distribution. They do not
  establish online strength, long-horizon stability, or the best rate after the model size changes.
- Persistent clipping can mean useful bounded updates or an ill-scaled optimizer. The short screens observed it but
  did not isolate the best clip norm for the retained objective with auxiliary heads.
- The exact retained SGD schedule has no one-variable online counterfactual. Final reporting must present it as the
  training recipe, not as a universally optimal optimizer result.
- The matched-estimator terminal advantage cannot be allocated to SGD. Only about 30 Elo of the roughly 74-Elo
  cross-campaign gap appeared in the one-expansion policy instrument; the remaining approximately 44 Elo appeared
  with 64-search tree use and has no isolated explanation. An optimizer hypothesis should therefore be presented as
  the best-supported contributor, not a complete decomposition.
- Dynamic loss balancing, gradient accumulation, EMA/SWA, cosine decay for the retained recipe, and alternative
  optimizers remain proposals rather than completed investigations.

### Sources

- [Retained training configuration](../../../py/configs/production/chess-final-config.yaml)
- [Current optimizer and warmup implementation](../../../py/src/training/trainer/rank.py)
- [Optimizer construction and checkpoint restoration](../../../py/src/training/checkpoint/persistence.py)
- [Optimizer calibration within the INT8 replay study](../../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md)
- [Nesterov replay screen](../../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md)
- [Pre-fold schedule factorial](../../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md)
- [Post-fold learning-rate sweep](../../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md)
- [Fixed-teacher optimizer and schedule study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Historical training optimizations](../../history/optimizations/training.md)
- [Final evidence index](../../evidence/final-chess-20260923/README.md), including the checksum-covered TensorBoard
  archive (pending a committed causal-comparison derivation)

## Progressive candidate training and promotion

### Question

- How can early training exploit a smaller model's throughput, introduce additional capacity as progress slows, and
  determine whether a successor is ready without sacrificing the active model's progress?

### Current mechanism

- Candidate training starts only after the searched-strength curve meets the configured plateau condition. The
  detector uses a bias-corrected Elo exponential moving average with decay `0.90`, measures its gain across six
  observation intervals using seven retained EMA samples, and requires two complete below-threshold windows. The
  intended worthwhile-gain thresholds are 15 Elo/hour before starting the middle-sized candidate and 4 Elo/hour
  before starting the largest candidate.
- The active model always trains one 500-step quantum for each credit-ledger quantum. A non-active candidate trains
  an average of 1.5 quanta: whole quanta alternate one, two, one, two according to the outer training-quantum index.
  The active model never inherits this multiplier.
- Extra candidate quanta are outside the presentation-credit ledger. Credits fund the active model's one quantum;
  candidate catch-up adds wall-clock optimizer work against the already captured replay snapshot.
- The active model and its immediate successor train sequentially against the same immutable replay description and
  replay-source optimizer-step identity. Because replay sampling and augmentation are seeded by that identity, their
  first quanta use the same deterministic sample stream. When the multiplier supplies a second candidate quantum, it
  repeats that stream after the candidate's weights have advanced.
- A candidate has its own learning-rate clock. The canonical schedule decays linearly from `0.1` to `0.01` across 600
  candidate-local quanta; it is not read at the much later global training index. During the completed campaign this
  floor proved too low once the active model was also at 0.01, so a campaign-specific continuation raised the
  candidate floor to 0.03.
- Promotion is now based on head-to-head play. A candidate must score at least 0.48 in two consecutive completed
  paired matches against the active checkpoint. A failing score resets the count. A failed or cancelled evaluation
  is absence of evidence and does not reset it.
- Ordinary controller candidates start from seeded random initializations. Only the immediate successor trains;
  model sizes cannot be skipped. The completed campaign separately tested a manually grown successor whose initial
  function matched its parent; that path is documented in the network dossier and is not yet controller behavior.
- After every required model has completed its work, the state machine chooses the active model, persists private
  candidate state, and publishes one public checkpoint. Extra candidate quanta do not cause intermediate self-play
  publication.

### Evidence and rationale

- The state machine, fractional multiplier, own-clock learning rate, exact replay identity, match gate, and
  crash recovery have focused tests. This is strong mechanism evidence.
- The multiplier was added because a from-scratch larger candidate could not erase the active model's accumulated
  training deficit at one quantum per outer quantum. Averaging 1.5 gives it additional catch-up work while keeping
  active progress and credit consumption unchanged.
- The candidate multiplier initially contained a clocking defect: indexing the fractional schedule by candidate
  progress made it settle at two quanta per generation. It is now indexed by the outer generation and alternates one
  and two as intended.
- The former loss-EMA gate was invalidated empirically. Because the candidate received more optimizer work on the
  same replay distribution, its training loss was systematically advantaged and could reach parity while its playing
  strength remained far behind. It promoted a larger candidate that then lost about 270 Elo. Match-based promotion
  replaced the loss comparison rather than merely adjusting its threshold.
- The small-to-medium handoff has worked consistently across the completed runs: the small model supplies high early
  throughput, then the medium model takes over as the small model's capacity-limited progress slows.
- The medium-to-large handoff is unresolved. Independently initialized large candidates took an impractically long
  time to catch up and did not demonstrate a clear gain. The final function-preserving experiment removed the initial
  relearning deficit, but its limited continuation only established parity; it did not test whether longer training
  would turn the additional capacity into better generalization.

### Pitfalls and unknowns

- Training loss remains useful for optimization diagnostics, but cannot compare promotion candidates that receive
  unequal replay presentations. The failed gate is a transferable warning against treating replay fit as strength.
- Repeating the same deterministic batch stream in a multiplier-supplied second quantum improves comparability and
  recovery but reduces the unique rows seen during candidate catch-up.
- The exact 1.5 multiplier, catch-up schedule, 0.48 match threshold, and two-match confirmation do not have a
  fixed-model or alternative-controller counterfactual. They are current control semantics, not isolated Elo results.
- The large-model outcome cannot establish that capacity was unimportant. Insufficient continuation, the
  optimization dynamics of newly exposed units, self-play targets, and replay composition remain confounded.
- Final reporting must record candidate start, candidate-local optimizer steps, extra wall time, promotion comparisons,
  promotions or reversions, and which model produced each admitted replay interval.

### Sources

- [Current progressive configuration](../../../py/configs/production/chess-final-config.yaml)
- [Candidate-start, multiplier, and promotion state machine](../../../py/src/training/progressive.py)
- [Sequential candidate training and own-clock learning rate](../../../py/src/training/session.py)
- [Trainer replay identity and comparable objective resolution](../../../py/src/training/trainer/group.py)
- [Fractional multiplier and recovery tests](../../../py/test/test_progressive_model_sizing.py)
- [Function-preserving growth procedure](../../../py/tools/grow_checkpoint.py)

## Initialization, bootstrap-policy calibration, and deterministic seeding

### Question

- How can architecture comparisons and first self-play targets be trusted when random initialization can create a
  pathological policy prior or nominally matched runs can start from different tensors?

### Alternatives and mechanisms

- Uncalibrated default initialization was initially allowed to determine the generation-zero policy distribution.
  That proved unsafe when new policy heads and trunks emitted extremely concentrated logits.
- The retained initialization procedure creates 20 candidates from a deterministic seed sequence. It rejects a
  candidate if its top action is too concentrated, its WDL entropy is too low, or its expected value is too extreme.
- The selected candidate's trainable policy scale is calibrated on 516 real positions until aggregate top-three
  policy mass is approximately `0.95`. This changes logit scale, not action ranking, trunk features, or WDL values.
- Before the first optimizer quantum, the coordinator performs another policy-health check against real replay data.
- TensorRT/QAT calibration is a separate operation. It measures activation ranges on 516 immutable evaluation
  positions and is refreshed for deployment each model publication. It must not be described as bootstrap-policy
  calibration.

### Evidence and results

- The direct-plane and attention experiments exposed random initial logit standard deviations far outside the proven
  dense-head scale. The resulting policy was nearly one-hot before learning and changed the self-play distribution
  from the first games. Scaling each architecture to the same top-three mass fixed that specific bootstrap defect.
- A later regression audit found that the configured random seed was applied only inside trainer processes, after
  the generation-zero model had already been constructed. Separate runs with the same configured seed therefore
  started from different weights even though ranks within each run agreed.
- Git history contains two distinct repairs: seed trainer construction, then seed checkpoint-zero construction
  itself and test same-seed tensor identity, identical calibration, and different-seed divergence.

### Decision rationale

- Initialization health is part of experimental validity, not an architecture improvement. A bad random prior
  changes which games are generated, so later losses cannot recover the missing matched counterfactual.
- Candidate screening and scale calibration are retained correctness measures. They reduce catastrophic starts but
  do not make different seeds equivalent and do not prove that the selected seed is representative.

### Pitfalls and unknowns

- Equal aggregate top-three mass does not equalize move ranking, value initialization, or gradient geometry.
- Early architecture conclusions produced before bootstrap repair remain incident evidence, not clean ablations.
- The report should not credit Elo to the 20-candidate selection without a controlled online comparison.
- Exact determinism still depends on CUDA kernels, sampling order, and native self-play randomness after checkpoint
  construction. The repaired seed guarantees the initial tensors, not a bit-identical full training trajectory.

### Sources

- [Current initialization and policy calibration configuration](../../../py/configs/production/chess-final-config.yaml)
- [Checkpoint-zero construction](../../../py/src/experiment/run.py)
- [Bootstrap-prior study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Initialization and warmup regression audit](../../analysis/v35-v42-regression-audit-20260913.md)
- [Trainer-construction seeding repair](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/d52949c7)
- [Checkpoint-zero seeding repair](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/99247375)

## Replay capacity, freshness, and unique information

### Question

- How much history should remain trainable as the policy improves, and what does a larger replay capacity actually
  buy?

### Alternatives and mechanisms

- A small recent sliding window discards weak early targets quickly but concentrates training on a narrow policy
  distribution.
- A large fixed window preserves variety but initially contains no more information than has actually been generated;
  it can also subsidize stale targets if data production does not keep pace.
- The retained design preallocates a 20-million-row physical memory map while growing only the logical FIFO capacity:
  0.6, 1.2, 2.0, 2.8, 4, 6, 8, 12, 16, and finally 20 million rows. Expanding logical capacity does not rewrite the
  store.
- Capacity, occupancy, age, and uniqueness are different. Capacity is a limit; occupancy is current live rows; age
  measures policy staleness; unique information is reduced by duplicate or highly correlated positions.
- An older shard-based replay implemented idle compaction and compatible-target deduplication. Duplicate states were
  aggregated, policies and scalar targets averaged, hard conflicting WDL targets kept separate, and capped
  square-root multiplicity became a loss weight. That architecture was superseded by the fixed-layout columnar store.
  Current replay does not perform global deduplication or compaction.

### Evidence and results

- A mature earlier training archive contained 187.5 million admitted positions but retained only a bounded live
  window. It showed that strength could keep improving after fixed-dataset policy accuracy largely saturated, while
  larger models and deeper search reduced fresh-position throughput. This motivates tracking age and fresh-data rate,
  not merely nominal capacity.
- The old storage benchmark compared 5,000 producer shards with 25 compacted containers over 2.5 million unique
  positions. Loader throughput rose from 3,943 to 32,579 samples/s, an 8.26-fold systems improvement. Row-read
  amplification increased, and the result measured I/O layout rather than model strength.
- The exact retained capacity schedule has no fixed-capacity online control. Its rationale is data availability and
  distribution breadth, not an isolated measured multiplier.

### Decision rationale

- Staged logical growth was retained because reserving the terminal window before enough data exist adds no diversity,
  while later growth reduces concentration on only the newest policy.
- The current store trades deduplication and immutable-container compaction for a simple bounded circular memory map,
  direct column gathers, and clearer crash behavior.
- Any future capacity increase should be expressed in hours of fresh data and age percentiles. A larger integer alone
  is not evidence of a better replay distribution.

### Pitfalls and unknowns

- The final archive must report occupancy, source-age percentiles, unique-state estimates, duplicate multiplicity,
  and sampling probability by age. None can be inferred from configured capacity alone.
- Increasing capacity without increasing fresh generation can make the mean target older and slow policy iteration.
- Current replay stores source generation and timestamp but does not use either for recency weighting.
- Global deduplication, recency weighting, and age-aware eviction were discussed but are not retained experiments.
- A newly added administrative pruning tool can remove rows produced by a known invalid model interval. It is a
  recovery boundary, not a general sampling or deduplication mechanism.

### Sources

- [Current replay configuration](../../../py/configs/production/chess-final-config.yaml)
- [Columnar store implementation](../../../py/src/replay/store.py)
- [Current replay-system description](../../system/replay-and-data.md)
- [Training-dynamics and fresh-data audit](../../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md)
- [Historical mature-loader and compaction benchmark](../../benchmarks/replay-loader-20260724/README.md)
- [Historical deduplication/compaction implementation](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/ebaef128)
- [Targeted invalid-row pruning mechanism](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/1109739e)

## Replay reuse and presentation-credit accounting

### Question

- How many optimizer presentations should each newly admitted row fund, and how can statistical reuse be separated
  from wall-clock schedule pace?

### Mechanism

- Replay ratio is defined as training presentations per newly materialized row. With ratio four, each durable row
  earns four credits.
- One optimizer quantum consumes `2,048 × 500 = 1,024,000` presentation credits. At ratio four, 256,000 newly
  appended rows fund a quantum.
- Credits are earned from the replay store's monotonic `total_appended_rows`, after append and flush. FIFO eviction
  does not retract credits, because those presentations were funded when the row became durable.
- The ledger persists earned credits, consumed credits, completed optimizer steps, and the active checkpoint. A
  quantum commits exactly once with its checkpoint. On restart, exactly one complete next checkpoint may be adopted
  if the ledger has sufficient unconsumed credit.
- The configured ratio therefore changes both statistical reuse and how quickly optimizer quanta, model publication,
  evaluation, capacity schedules, and other optimizer-indexed schedules advance in wall time.

### Alternatives and evidence

- Successful prior training used a ratio of eight. Short matched controls compared ratios four, 6.25, and eight on
  the same small-model platform.
- The preserved decision record says ratios four and 6.25 matched at every shared evaluation boundary while their
  update rates differed by the ratio, suggesting an update-count/freshness cancellation over that short horizon. A
  ratio-eight arm was launched to widen the separation.
- The terminal retrospective confirms that ratios four, 6.25, and eight remained matched at every shared boundary
  in the available controls, but all arms ended within 90 minutes. The completed campaign used ratio four and twice
  the replay capacity of the predecessor, so it exposed roughly twice as many fresh positions per optimizer step;
  because ratio and capacity moved together and no long control exists, neither setting receives causal credit for
  the final matched-estimator gain.
- The raw curves and complete result bundle for those three short controls are not preserved under the benchmark
  directory. The conclusion therefore has weaker provenance than the checked-in replay and training benchmarks.
- The retained ratio is four. That is a deliberate fresh-data bias, but the repository has no long matched online
  experiment proving four is the strength optimum.

### Decision rationale

- Lower reuse buys more unique data per optimizer presentation when actor throughput can supply it. Higher reuse buys
  more updates per row and advances optimizer-indexed schedules faster.
- Because the two effects are coupled in the credit runtime, the report must not say that a ratio arm changed only
  sample efficiency. The controlled variable also changed wall-clock publication and evaluation cadence.
- Report both configured ratio and empirical presentations per admitted unique row. Rejection, duplication,
  interruption, and capacity turnover can make the latter more informative.

### Pitfalls and unknowns

- A replay ratio is not an epoch count: batches sample across the live FIFO, rows can repeat across steps, and rows
  can be evicted before receiving an equal number of presentations.
- More presentations can improve frozen-replay loss while reducing the rate at which new policy targets enter the
  learner.
- The final archive needs fresh rows per second, quanta per hour, replay age, and effective unique presentations.
- The short matched controls need raw recovery before their numerical curves can be publication claims.
- “More fresh data per step” is a mechanistic description of the configured ratio, not evidence that the changed
  replay regime improved Elo. The terminal comparison also changed optimizer, objective weight, and inference path.

### Sources

- [Credit-ledger implementation](../../../py/src/training/credit_ledger.py)
- [Current replay and credit configuration](../../../py/configs/production/chess-final-config.yaml)
- [Credit and snapshot description](../../system/replay-and-data.md)
- [Training dynamics with empirical ratio and admitted-data rate](../../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md)
- [Preserved short-control decision record](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/713d3439)
- [Matched-estimator and causal-bound source note](evaluation-and-pitfalls.md#retrospective-matched-estimator-audit)

## Model-publication cadence and target freshness

### Question

- How often should improved weights replace the self-play model, given that more frequent refresh makes generated
  targets more current but checkpoint export, deployment conversion, actor refresh, and lost batch efficiency cost
  wall time?

### Alternatives and mechanisms

- A historical experiment published every 100 optimizer steps. This reduced the maximum optimizer lag of self-play
  data but repeated the complete publication path five times as often as the retained 500-step quantum.
- Less frequent publication amortizes checkpoint writing and inference-artifact preparation, lets trainers and actors
  run longer between transitions, and produces larger policy changes between refreshes.
- The retained lifecycle publishes at most one public model after each 500-step credit-funded outer quantum. In
  progressive training, private active and candidate checkpoints may be written while the quantum is in progress,
  but self-play sees only the selected active checkpoint after every required model has finished.
- Worker refresh is in place at a desired-state boundary. A worker finishes its current native batch, adopts the new
  deployment artifact, and discards retained trees whose priors and values belong to the previous model.
- Replay rows record source model and timestamp, but current sampling does not explicitly favor or down-weight rows by
  publication age. Cadence influences freshness through generation, not through an age correction in the learner.

### Evidence and result

- The historical project ledger records the 100-step cadence as implemented and rejected: publication overhead
  exceeded the observed freshness benefit. No raw timing bundle or matched strength curve for that experiment is
  preserved, so this is a decision record rather than a quantitative result suitable for an effect-size claim.
- On the current TensorRT path, a measured training interval attributed approximately 7–10 seconds per outer quantum
  to checkpoint publication, refit, and activation. Another backend benchmark measured a real two-worker refresh at
  `0.220` seconds; that refresh number excludes the whole trainer checkpoint and publication pipeline.
- Older direct-refresh benchmarks established that in-place inference replacement can preserve allocated tree
  structures without a transient memory leak. The current worker nevertheless clears semantic tree state on model
  transition so old evaluations are not mixed with new weights.

### Decision rationale

- The retained 500-step boundary is a compromise between freshness and fixed publication cost. The negative 100-step
  result justifies rejecting “publish as often as possible”; it does not establish 500 as a universal optimum.
- Publication cadence is coupled to replay reuse. Changing steps per publication while holding presentation ratio
  fixed changes the number of fresh rows per policy iteration, and changing reuse changes how quickly a 500-step
  publication is funded.

### Pitfalls and unknowns

- “Model refresh latency” and “publication overhead” are different. The former can time only an already-built artifact
  swap; the latter includes checkpoint serialization, export/refit, validation, synchronization, and activation.
- Very frequent publication can also destabilize target stationarity even if export were free. That hypothesis was
  considered but was not isolated from overhead in the recorded experiment.
- The old 100-step result predates the final TensorRT/QAT and progressive-candidate pipeline. Its qualitative lesson is
  relevant, but its magnitude cannot be transferred.
- A future cadence comparison must report publication critical-path time, actor model-age lag, fresh rows per public
  model, discarded trees or in-flight work, and strength per wall-clock hour.

### Sources

- [Historical publication-cadence decision record](../../history/historical-research-backlog-20260822.md)
- [Current credit and quantum configuration](../../../py/configs/production/chess-final-config.yaml)
- [Progressive publication state machine](../../../py/src/training/session.py)
- [Self-play model refresh boundary](../../../py/src/self_play/worker.py)
- [Measured current training/publication decomposition](../../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md)
- [Native TensorRT publication and refresh benchmark](../../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md)
- [Historical in-place refresh benchmark](../../benchmarks/model-refresh-20260723/README.md)

## Row admission, row selection, and loss weighting

### Question

- Where can the system emphasize or exclude a position, and what claim is justified for each mechanism?

### Mechanisms

- **Admission:** the materializer reconstructs a completed trajectory and creates one primary row for each stored
  search observation. Random opening-prefix moves and replayed restart prefixes have no search observation and do not
  become rows. A final unplayed cut-position search can become a row even though it selects no action.
- **Sparse policy retention:** up to 60 actions are stored, ordered by visit count with action ID as deterministic
  tie-break. Discarded visit mass is measured; the retained mass is normalized when the dense batch target is built.
- **Selection:** the current sampler mixes 30% uniform draws with 70% policy-surprise draws. Surprise is
  `KL(search visit policy || raw network prior)` and is capped at `2.0`. Sampling is without replacement within one
  global batch; the same row may reappear in later steps.
- **Weighting:** every observation stores a positive sample weight. In the objective, selected weights are divided by
  their batch mean and multiply primary policy, WDL, and eligible auxiliary losses. The retained self-play recipe
  emits weight `1.0`, so current surprise priority changes draw probability rather than loss magnitude.
- **Credits:** all admitted rows earn the same configured presentation credits. Surprise and sample weight do not
  change credit issuance.

### Alternatives and evidence

- Uniform sampling remains the fallback when surprise mass is zero and contributes a permanent 30% exploration
  mixture otherwise.
- Historical duplicate aggregation used multiplicity-derived sample weights. That validates the typed weighting path
  but does not prove that arbitrary weighting improved chess strength.
- TD-error priority, direct policy-loss priority, recency weighting, and importance-sampling correction were proposed.
  They are not completed current experiments.
- The retained surprise sampler is part of a successful bundle, but there is no isolated online Elo ablation.

### Decision rationale

- Policy surprise targets positions where search most revised the prior and therefore where the network may have the
  most to learn. The cap prevents a few pathological rows from dominating, and the uniform mixture preserves broad
  coverage.
- Sampling and weighting must stay separate in publication language. A row drawn more often is not necessarily given
  more influence each time, and the retained recipe uses unit row weights.

### Pitfalls and unknowns

- Sampling by raw surprise changes the empirical objective unless corrected by inverse probability. The current
  implementation intentionally does not apply such a correction; it trains on the prioritized distribution.
- Surprise can correlate with search noise, shallow targets, model age, game phase, or rare legal-action counts. The
  final archive should plot these relationships before describing it as pure “difficulty.”
- Batch-mean weight normalization keeps average scale stable but makes the influence of one weighted row depend on
  the other rows in its batch.
- No current evidence isolates the surprise cap, uniform fraction, or any non-unit sample-weight schedule.

### Sources

- [Native policy-surprise calculation](../../../cpp/src/search/SearchExecutor.hpp)
- [Replay admission and target materialization](../../../py/src/replay/materialization.py)
- [Replay sampling implementation](../../../py/src/replay/batch_loader.py)
- [Weighted objective implementation](../../../py/src/training/objective.py)
- [Current sampling and weight configuration](../../../py/configs/production/chess-final-config.yaml)
- [Sample-stream audit](../../analysis/v8-training-data-comparison-20260826.md)

## Value-target construction: outcome discount, root-value blend, and search-backup discount

### Question

- How should a value target combine the eventual game result with information from search, and which of the project's
  three similarly named value mechanisms acts at each boundary?

### Three distinct mechanisms

1. **Replay outcome discount (`0.998` per remaining ply).** During materialization, the terminal WDL is oriented to
   the player at the sampled position. It is then blurred toward uniform by `0.998 ** remaining_plies`. This changes
   the durable outcome target stored in replay. The exponent uses distance from that observation to the end of its
   generated game and the schedule value at the observation's source-model index.
2. **Optimizer-time root-value blend (zero to `0.1`).** Replay separately stores the search root scalar for each
   observation. At loss calculation, the scalar is converted to a soft WDL distribution: its signed magnitude becomes
   win or loss mass and the remaining probability is divided equally among win, draw, and loss. Linear interpolation
   then blends the discounted terminal WDL with that search-derived WDL. The blend is zero through model index 50,
   rises linearly, and reaches `0.1` at index 110. It is resolved for the replay-source progress used to compare active
   and candidate training, so both models in one outer quantum receive the same objective.
3. **Search-backup discount (`0.99` per tree ply).** Native search discounts values while backing them through the
   tree. It changes root values, child Q values, selection, and potentially the visit-policy target before a game is
   written. It does not directly blur the stored terminal WDL and is not the optimizer-time interpolation coefficient.

For a stored terminal target `z`, stored root scalar `r`, remaining game distance `d`, and optimizer-time blend
`alpha`, the training value target is conceptually:

```text
discounted_outcome = uniform_blur(z, 0.998 ** d)
training_wdl = (1 - alpha) * discounted_outcome + alpha * scalar_to_wdl(r)
```

The search value `r` was itself produced by a tree whose backups use `0.99` per ply. That provenance does not make
the two discounts interchangeable.

### Alternatives and evidence

- Pure terminal-outcome training is recovered when the blend is zero. Historical recipes used larger blends, while
  the retained schedule delays and caps search-derived supervision at 10%.
- The project has implementation and successful-run evidence for the blend but no isolated long online comparison of
  zero versus the retained schedule.
- The cut-position benchmark supports searched root values as useful bootstraps: they beat material targets on Brier
  score and cross-entropy in both measured cohorts. It does not isolate the ordinary optimizer-time 10% blend.
- The late-game incident showed why target provenance matters. A shallow cut-search value propagated across a whole
  game can poison replay even when the same mathematical blending machinery is sound for ordinary full-search roots.

### Decision rationale

- Early search values come from a weak network and weak tree, so the retained blend begins at zero. Later, a small
  search component can provide denser information than a single terminal outcome while leaving the observed result as
  90% of the target.
- Per-ply outcome discount expresses less certainty about assigning a distant final result uniformly to every earlier
  decision. Search-backup discount instead regularizes long tree lines. Both are retained, but neither constitutes
  evidence for the root-value blend's causal strength.

### Pitfalls and unknowns

- `scalar_to_wdl(0)` is uniform thirds, not a confident draw. A scalar search value cannot preserve the search's full
  WDL decomposition.
- Blending a stored root value makes the target partly self-referential. Calibration drift or shallow-search bias can
  reinforce the network, which is why schedule, search depth, and cut handling must be reported together.
- The root value and terminal WDL are not independent labels. Search uses the same network and the game was generated
  by the searched policy.
- Final reporting should plot terminal-target entropy, root-value calibration, blend coefficient, and value loss over
  training. No numerical Elo gain should be assigned to the `0.1` blend without a controlled arm.

### Sources

- [Current value schedules](../../../py/configs/production/chess-final-config.yaml)
- [Outcome discount during replay materialization](../../../py/src/replay/materialization.py)
- [Root-scalar conversion and optimizer-time blend](../../../py/src/training/objective.py)
- [Objective resolution for chess training](../../../py/src/games/chess/training.py)
- [Native search-value backup](../../../cpp/src/search/SearchExecutor.hpp)
- [Cut-position value benchmark](../../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md)
- [Historical rationale for hybrid value targets](../../history/optimizations/training.md)

## Random openings and opening diversity

### Question

- How can self-play avoid repeatedly traversing the same opening while retaining legal, unbiased game states?

### Mechanism and alternatives

- Half of retained-recipe games start after a uniformly sampled count from zero through eight random legal plies.
  At each prefix ply, the action is uniform over currently legal actions. A prefix that reaches a natural terminal is
  discarded and redrawn.
- The random prefix is part of the recorded action history but is not searched and does not itself produce training
  rows. Search begins from the resulting position.
- The other half of starts request an archived restart state. There are no separately weighted ordinary-start games,
  but drawing zero random plies produces the ordinary initial position with probability one ninth within the random
  half.
- An empty restart archive falls back to the non-restart mixture, which in the retained recipe means a random opening.

### Evidence and rationale

- The mechanism is implemented, tested, and retained. It provides cheap opening diversity and ensures the replay
  stream is not dominated by a deterministic initial trajectory.
- No matched online experiment isolates zero-to-eight-ply random openings from restart states or from ordinary starts.
  The report should therefore describe the mechanism and motivation, not assign an Elo gain.

### Pitfalls and unknowns

- Uniform random legal moves are not a balanced opening book. They can enter strategically poor positions and their
  position distribution depends strongly on the maximum prefix length.
- Prefix actions do not receive direct policy supervision. Their benefit is the downstream positions they expose.
- Final reporting should include the realized start mixture, prefix-length histogram, early termination/redraw rate,
  and fallback rate from empty restart archives.

### Sources

- [Self-play start construction](../../../py/src/self_play/worker.py)
- [Start-parameter validation](../../../py/src/self_play/parameters.py)
- [Current start mixture](../../../py/configs/production/chess-final-config.yaml)

## Restart-state extraction, prioritization, reservation, and value correction

### Question

- Can complete games be redirected toward positions where search corrected the network and where an untried plausible
  continuation can add useful data?

### Mechanism

- Each self-play process owns a local SQLite archive. Only observations from completed games are considered.
- A position is eligible when at least 15 plies remained in the source game, absolute search-root value is at most
  `0.8`, and the smallest prefix of visit-ranked actions reaching 85% of root visit mass contains two or three
  candidates.
- The action actually played in the source game is marked tried immediately. The other candidate actions are branch
  opportunities.
- Position priority is the search value correction, defined as half the absolute difference between searched root
  value and raw network expected value. Seventy percent of reservations use rejection sampling proportional to the
  square root of correction relative to the archive maximum; 30% select a position uniformly.
- Once a position is chosen, the smallest-ID untried candidate is reserved transactionally and forced as the first
  action after reconstructing the prefix. Reservation prevents multiple future games from claiming the same branch.
  When all candidates are tried, the position is evicted.
- The archive is capped at 50,000 positions and 40 model-publication intervals of age. Oldest positions are evicted
  on capacity overflow. Archive state and counters survive worker restarts.
- A restart prefix, like a random-opening prefix, is reconstructed but not admitted as new training rows. New search
  observations begin at the archived state; the forced candidate changes the trajectory after that search.

### Alternatives and evidence

- Purely uniform archived starts were available through the mixture. The retained square-root priority deliberately
  softens the raw correction signal rather than always taking the maximum.
- Earlier Go-Exploit and regret-based proposals motivated the design, but this implementation does not train a regret
  network and does not use future game outcome to predict regret.
- Policy-surprise replay selection is separate: it changes which existing row trains the network. Restart priority
  changes which future games and branches are generated.
- The complete mechanism is production code with persistence and telemetry, but it lacks a one-variable online Elo
  ablation and a preserved benchmark of diversity gained per restarted game.

### Decision rationale

- Candidate visit-mass filtering chooses plausible alternatives rather than arbitrary legal moves. Marking the played
  branch tried focuses new compute on coverage the source game did not provide.
- Value correction provides a cheap “network was surprised by search” signal already available at generation time.
  The uniform component and square root prevent an extreme correction from monopolizing starts.
- Age and capacity bounds keep the archive relevant to the current policy and operationally bounded.

### Pitfalls and unknowns

- “Difficult state” is an interpretation of value correction, not ground truth. Large disagreement may indicate noisy
  search, calibration drift, or a tactical correction.
- Candidate selection after choosing a position is deterministic by action ID, not weighted by visits. That should be
  stated accurately if the report explains branch reservation.
- The archive is per worker, not globally shared, so candidates can be duplicated across workers.
- Final evidence should report eligibility rate, archive occupancy, priority distribution, uniform/prioritized split,
  reservations, exhausted/expired/capacity evictions, empty fallbacks, and downstream state diversity.
- The independent strength contribution of restart starts, their 50% mixture, the `0.8` value filter, and square-root
  priority is unresolved.

### Sources

- [Restart archive implementation](../../../py/src/self_play/restart_archive.py)
- [Native value-correction calculation](../../../cpp/src/search/SearchExecutor.hpp)
- [Self-play reservation and forced-branch behavior](../../../py/src/self_play/worker.py)
- [Current restart configuration](../../../py/configs/production/chess-final-config.yaml)
- [Reference-recipe analysis](../../analysis/reference-recipes-for-a-compute-poor-run.md)
- [Mixed-start and priority implementation record](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/aaeab181)

## Resignation calibration and continuation games

### Question

- How can clearly lost games stop early without silently converting draws or wins into false losses and poisoning
  value targets?

### Alternatives and mechanism

- Fixed resignation thresholds save search but become unsafe as value calibration and playing style change.
- Disabling resignation is safe for labels but spends substantial compute playing decided positions.
- The retained calibrator audits candidate thresholds from `-0.99` through `-0.70` in steps of `0.01`. Twenty percent
  of games are designated continuation games at creation and can never resign.
- A candidate trigger requires both the root expected value and the best visited child's backed-up value to be at or
  below the threshold. Requiring both avoids resigning when the aggregate root is pessimistic but search found a
  credible escape.
- Continuation games supply counterfactual outcomes for every threshold they crossed. Capped/adjudicated games are not
  treated as reliable safety evidence for a natural terminal result.
- Over a rolling window of 2,000 triggered continuation games, a threshold needs at least 100 triggers and a one-sided
  95% binomial upper bound on false non-loss no greater than 2.5%. Production resignation is disabled before the
  configured calibration boundary.
- Relaxation toward a less negative threshold is limited to `0.01` per model publication. If evidence worsens, the
  policy can become more conservative immediately. Calibration state has an idempotent SQLite observation journal.

### Evidence and results

- A deliberately aggressive compute-node canary forced the continuation and persistence paths. A hypothetical trigger
  at ply five was recorded, but the game ended at the ply cap, so it was correctly excluded from safety evidence and
  resignation remained disabled. The canary validated mechanics, not a safe production threshold.
- Historical comparisons showed that raw resignation rate did not explain a severe conversion failure. Continuation
  assignment remained close to its configured probability, and a non-converting model actually resigned later than
  a converting reference.
- The retained 20% continuation rate and 2.5% safety ceiling were bundled with several late-game repairs. Their
  independent effect is not isolated.

### Decision rationale

- Permanent continuation games turn threshold calibration into an ongoing measured safety process rather than a
  one-time guess. Slow relaxation prevents one favorable window from making a large aggressive jump.
- Resignation remains a compute optimization with label-risk controls. The right publication claim is “calibrated and
  audited,” not “proved to make no false resignations.”

### Pitfalls and unknowns

- A continuation is assigned per game, not only after a trigger. The realized continuation-game fraction therefore
  should be compared with all games, while safety evidence uses the triggered subset.
- Search-root and best-child values share model and search errors; requiring both is not independent confirmation.
- The terminal archive must report the threshold path, evidence counts, false non-losses and confidence bounds,
  trigger ply, saved plies/search, actual resignations, and interactions with restart starts.
- No completed experiment measures the Elo or wall-clock gain of the calibrated policy against resignation disabled.

### Sources

- [Calibrator, journal, and threshold selection](../../../py/src/self_play/resignation.py)
- [Trigger and completion behavior](../../../py/src/self_play/worker.py)
- [Current resignation configuration](../../../py/configs/production/chess-final-config.yaml)
- [Resignation audit canary](../../benchmarks/resignation-audit-canary-20260723/README.md)
- [Sample-stream and continuation audit](../../analysis/v8-training-data-comparison-20260826.md)
- [Endgame-conversion investigation](../../analysis/chess-conversion-investigation-20260826.md)

## Cut-game targets, remaining-length censoring, and late-game poisoning

### Question

- What target should a game receive when a wall-clock-motivated ply cap stops it before a natural terminal result, and
  how can the cap avoid teaching the model that winning but unconverted positions are draws?

### Alternatives and mechanisms

- Alternatives investigated for a cut position included a material heuristic, raw network value, searched root value,
  previous-ply searched value, sharpened material transforms, and blends.
- The retained worker performs one search at the cut position itself and converts that root scalar to a WDL target.
  This observation is stored even though it chooses no move. The target then belongs to the actual side to move and
  requires no sign workaround.
- All admitted observations from the cut game receive that final WDL with the normal per-ply training discount.
- Remaining game length has no known label for a cut game. Its auxiliary target is marked ineligible on every row
  from that game rather than pretending the cap equals the natural remaining length.
- The maximum game length grows from 150 to 250 plies as training advances. This limits early cost while allowing
  stronger later models more room to convert.

### Evidence and results

- A dedicated study played games beyond the ordinary 150-ply cut and compared candidate predictors with the eventual
  natural result. At the earlier measured checkpoint, searched root value achieved Brier `0.444` and cross-entropy
  `0.756`, versus `0.491` and `0.851` for material divided by 39. At the later checkpoint it achieved `0.193` and
  `0.374`, versus `0.388` and `0.707` for the material heuristic. Search-root sign accuracy exceeded 98% in both
  cohorts; calibration, not sign, was the meaningful difference.
- Sharpened material could fit one cohort and be badly wrong in another. The searched value was the more stable
  bootstrap and required no separate hand-tuned material scale.

### Incident study: late-game target poisoning

- An early recipe switched to cheap searches before the game cap. Because only expensive-search observations were
  then admitted, about 14.4% of all played plies in the earliest window were excluded—and they were specifically the
  last plies of the longest games.
- Cut games simultaneously took their final value from an 81–136-simulation cheap search. That one shallow estimate
  was propagated to every admitted row in the game, affecting roughly 36–38% of rows in the worst window.
- The bad rows later aged out of replay, but the network weights retained the learned damage. Current-replay analysis
  therefore looked healthy even while play still produced drawn-out, unconvertible wins.
- The resulting loop was self-reinforcing: absent full-search endgame rows produced a weak late-game policy; cheap
  searches driven by that policy played nearly randomly and failed to finish; the cutoff heuristic then supplied the
  same poor value supervision instead of a natural result.
- The first repair replaced that heuristic with one full search at the cut position and used its root value. The later
  repair removed the cheap-search tail and restored properly searched endgame positions to replay. A more
  conservative cap schedule, per-ply target discount, and continuation changes were bundled around this work, so no
  single component can receive an isolated Elo effect.

### Decision rationale

- A searched root at the actual cut position is retained because it beat the material heuristic on calibration and
  proper scoring rules, and because it avoids taking a value from a deliberately cheap preceding search.
- Censoring remaining length is required by target semantics. It is not optional regularization.
- This is one of the few places where temporal narration is necessary: the danger is the sequence “exclude deep
  policy rows, generate a shallow terminal bootstrap, propagate it backward, then let replay eviction hide the bad
  data while weights retain it.”

### Pitfalls and unknowns

- A cut position has no true terminal label at training time. The root value is still a model-derived bootstrap, not
  ground truth.
- The cut-value benchmark measured specific early checkpoints and search settings. Calibration should continue to be
  audited as network strength changes.
- Current target discount blurs the terminal WDL toward uniform with distance. Its exact `0.998` value was not isolated
  from the bundled conversion repair.
- Final reporting needs cut frequency, capped-game outcome estimates, remaining-length eligibility, and value
  calibration by game phase.

### Sources

- [Cut-position target benchmark](../../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md)
- [Conversion incident analysis](../../analysis/chess-conversion-investigation-20260826.md)
- [Replay sample-stream forensics](../../analysis/v8-training-data-comparison-20260826.md)
- [Current cut handling](../../../py/src/self_play/worker.py)
- [Target materialization and censoring](../../../py/src/replay/materialization.py)
- [Current cap and target configuration](../../../py/configs/production/chess-final-config.yaml)

## Auxiliary-target materialization and eligibility

### Question

- Which labels can be derived from a completed trajectory, and how are unavailable labels prevented from silently
  becoming zeros or fabricated targets?

### Mechanism

- The retained auxiliary heads are next-policy at a one-ply offset and remaining game length normalized by 400 plies.
- Next-policy uses the sparse visit distribution from the later search observation. It is eligible only when that
  exact later observation exists. Natural termination, resignation, cuts, or missing searches can make it ineligible.
- Remaining length is derived from the completed action sequence and is eligible for naturally resolved games. It is
  explicitly ineligible for every row of a cut game.
- Eligibility is stored beside each target in replay. The objective multiplies auxiliary loss by the eligibility mask
  and normalizes by eligible sample weight, so ineligible rows do not dilute the auxiliary mean.
- Primary and auxiliary policies share deterministic top-60 visit retention. A future policy can be retained for an
  auxiliary even if that future observation is not otherwise used as the current row's target.
- Other implemented target layouts include future search value, irreversible progress, and legal moves, but they are
  not part of the retained recipe. Proposed material, king-safety, uncertainty, control-map, and similar heads do not
  have completed production evidence.

### Evidence and rationale

- Overfit studies verified that the multi-head objective can learn its supplied targets. Replay forensics measured
  remaining-length censoring in real stores. These establish wiring and trainability, not an independent Elo gain.
- The owner confirms that completed runs used broader auxiliary-head bundles. Most were later disabled during a period
  with several simultaneous training problems to remove plausible sources of instability. This was precautionary
  simplification, not a planned auxiliary-head ablation, and no exact run/result bundle is currently identified.
- Trajectory-level materialization is retained because future-dependent labels cannot be constructed correctly from
  isolated rows at training time.

### Pitfalls and unknowns

- A cheap or missing future search can remove the next-policy label from an otherwise valid primary row. Eligibility
  therefore depends on data-generation semantics.
- Auxiliary weights `0.15` and `0.1` were not isolated in a long online ablation, and gradient interaction with the
  shared trunk remains unquantified.
- Neither their removal nor the later retention of next-policy and remaining-length proves that an auxiliary head
  helped or harmed playing strength. Report the debugging rationale without assigning a causal result.
- The final archive should report eligibility by target, termination reason, game phase, and model stage.

### Sources

- [Target layouts](../../../py/src/training/targets.py)
- [Trajectory materialization](../../../py/src/replay/materialization.py)
- [Masked auxiliary objective](../../../py/src/training/objective.py)
- [Current auxiliary configuration](../../../py/configs/production/chess-final-config.yaml)
- [Multi-head overfit study](../../benchmarks/chess-overfit-rtx3090-20260819/README.md)
- [Replay target audit](../../analysis/v8-training-data-comparison-20260826.md)

## Reanalysis and target freshness

### Question

- Is stale replay better refreshed by re-searching old positions with the current model, or by spending the same
  search compute on new games?

### Implemented historical design

- An older replay system stored lossless starting FENs and complete move histories so positions could be reconstructed.
- After publication, one designated self-play worker synchronously re-searched a bounded fraction of a recent producer
  payload using the new model and the full self-play search budget.
- Immutable sidecars were bound to the source payload hash. They overrode policy visits and search-root value during
  replay decode while leaving terminal outcome and material supervision unchanged.
- Disjoint sidecars accumulated; if a row was refreshed more than once, the newest model's target won. Compaction
  materialized active overrides before retiring source payloads.
- Reanalysis did not issue new presentation credits because it refreshed targets rather than admitting new positions.

### Evidence and decision rationale

- The implementation, persistence, and tests are preserved in Git. There is no controlled result showing improved
  learning or Elo per unit search.
- The design was removed when replay ownership moved to the fixed-layout columnar store. Current rows do not preserve
  the trajectory provenance needed by that sidecar implementation.
- Reanalysis is therefore superseded infrastructure, not a negative efficacy result. It should not be described as
  “tried and found harmful.”
- Revival should be gated on a measured staleness problem: replay age, current-versus-source policy divergence, and
  refreshed-target value per search compared directly with fresh self-play.

### Pitfalls and unknowns

- Synchronous work inside one publication acknowledgement can delay model refresh for that worker.
- Refreshing only producer payloads and not already compacted containers biased which rows could receive updates.
- Reanalysis changes target freshness without adding state diversity or terminal outcomes. Fresh games add all three.
- No current replay schema or atomic override owner has been designed for revival.

### Sources

- [Historical reanalysis and deduplication record](../../history/v10-training-quality-implementation.md)
- [Preserved reanalysis implementation commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/ba8b1fd7)
- [Reanalysis decision method](../../history/historical-research-backlog-20260822.md)

## Synchronous quanta, self-play overlap, and the boundary with full asynchrony

### Question

- How much actor work should continue while all GPUs train, and is the runtime accurately described as asynchronous?

### Mechanism and alternatives

- Training is credit-gated into 500-step quanta. Checkpoint publication and worker model refresh occur at coordinated
  boundaries.
- Before a quantum, the coordinator requests a subset of actors to pause but does not wait for a global worker barrier.
  Remaining actors may continue self-play while all eight trainer ranks use the GPUs.
- The retained topology has four self-play processes per GPU and requests two per GPU to pause during training. This
  is overlapped self-play and optimization, not fully asynchronous AlphaZero.
- Fully pausing actors maximizes trainer rate. Leaving all actors running maximizes concurrent search but slows every
  synchronized data-parallel rank. Intermediate balanced assignments were measured.
- Fully asynchronous training, continuous publication, and actors independently consuming arbitrary model ages were
  proposed but not implemented.

### Evidence and results

- In a controlled late-search workload, leaving zero, eight, sixteen, or thirty-two actors running produced trainer
  rates of 25,275, 21,492, 17,139, and 9,210 samples/s. Concurrent search increased from none to 505,509, 605,762,
  and 741,508 searches/s.
- Converting both streams into estimated quantum time produced only about a 5% spread, mildly favoring more concurrent
  actors in that regime. In an earlier, cheaper-search regime, the spread was only 2.3%. The optimum is workload
  dependent.
- The first measurement accidentally kept consecutive actor IDs, placing load only on the first GPUs. Because DDP is
  limited by its slowest rank, this produced a false 2.53-fold apparent advantage. Round-robin per-GPU placement
  corrected the result.

### Decision rationale

- Partial overlap was retained because it converts otherwise idle actor capacity into fresh search while avoiding the
  severe trainer slowdown of leaving every actor active.
- Coordinated quanta keep credit accounting, immutable replay snapshots, optimizer recovery, and checkpoint identity
  auditable. The project did not establish that the additional complexity of fully asynchronous learning would repay
  these lost invariants.

### Pitfalls and unknowns

- A topology benchmark must balance actors across devices; total worker count is insufficient.
- Search saved or generated during the trainer interval matters only if it shortens the critical path to the next
  credit-funded quantum.
- The retained two-paused-per-GPU setting is operational, not a universal optimum. Visit depth, model size, batch
  efficiency, and GPU topology can reverse the ranking.
- Final reporting needs trainer duty cycle, actor pause latency, search produced during quanta, backpressure time, and
  actual quantum wall time.

### Sources

- [Pause/overlap benchmark](../../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md)
- [Coordinator state machine](../../../py/src/training/coordinator.py)
- [Actor supervision and asynchronous pause protocol](../../../py/src/training/self_play_group.py)
- [Current process topology](../../../py/configs/production/chess-final-config.yaml)
- [System interaction dossier](system-architecture-and-figure-dossier.md)

## Replay storage, materialization, memory mapping, prefetch, and recovery

### Question

- How can thousands of concurrent games become validated, durable, high-throughput batches without a corrupt game,
  deep inbox, process crash, or Python object layout stalling training?

### Storage and materialization mechanism

- Self-play writes a typed completed-game JSON to a temporary path, flushes it, and atomically renames it into an
  inbox.
- A bounded dispatcher renames at most 4,096 inbox files per pass into eight stable per-worker directories. Rename on
  the same filesystem gives each game one owner without a shared claim queue or repeated hashing.
- Each long-lived materializer consumes its directory in counter order, up to 32 games and approximately 16 MiB of
  source JSON per shard. It reconstructs trajectories, validates legality and terminal semantics, materializes
  future-dependent targets, and writes typed columnar arrays.
- Shard data is flushed first and its manifest is written last. Identity derives from layout digest, worker, and
  counter range, so restart after sealing but before source deletion adopts the existing shard rather than duplicating
  it.
- One bad source is quarantined and the other games still seal. A rolling 512-game window aborts the run if rejection
  exceeds 5%, preventing silent total data loss. Sealed staging is bounded at 96 shards.
- The live store is one preallocated schema-checked circular memory map with a 64-KiB header and fixed column slabs.
  Appends copy each column into at most two ring slices. Logical FIFO capacity can grow without changing file layout.
- Cross-worker completion order is approximate; order within one worker directory is exact. Sampling does not depend
  on global chronological order, while source generation and timestamp remain stored for telemetry.

### Batch construction and prefetch

- At a training boundary, the coordinator captures replay path, head, size, logical capacity, and schema while holding
  the replay snapshot lock. Every rank opens the same mapping read-only and verifies that the snapshot still matches.
- One global seeded sample order is generated from sampler seed and source optimizer step. Each rank receives a
  non-overlapping local slice, so DDP does not duplicate-pad an incomplete batch.
- Packed state planes are decoded in vectorized NumPy operations. Sparse policies and legal actions are gathered in
  bulk, augmented, and scattered into the dense targets consumed by the objective.
- One CPU preparation thread per rank fills a bounded prefetch queue of depth four. Pinned host slots and one shared
  CUDA transfer stream support nonblocking copies without leaking allocator arenas across quanta.

### Evidence and results

- The earlier shard loader achieved only 3,943 samples/s before compaction and 32,579 after compaction, showing that
  file-open layout could bottleneck a 22,599-sample/s trainer.
- The production DDP path with real replay, deterministic non-overlapping partitions, decoding, pinning, transfer, and
  synchronization sustained 22,599 samples/s. Under simultaneous self-play contention it sustained 15,026 samples/s.
- Later vectorization, mapped columns, and prefetch were implemented to remove Python row objects and batch assembly
  from the optimizer critical path. Their tests establish ordering and ownership; individual commits are systems
  evidence rather than chess-strength evidence.
- The per-worker materialization redesign followed a production wedge in which a deep inbox made queue scans, hashing,
  and invalidation scale with backlog. The new dispatcher has bounded per-pass work and no head-of-line shard sequence.

### Recovery boundaries

- A dead materializer is restarted against its durable worker directory. Unconsumed files remain; an already sealed
  deterministic shard is adopted.
- On startup, sealed staging shards are appended normally, worker counters are reseeded, and directories belonging to
  removed workers are returned to the inbox.
- Training credit is reconciled only after a replay append and flush. This prevents the ledger from funding optimizer
  work with rows that are not durable.
- In-flight self-play games are persisted during orderly pause/shutdown and can resume with their action history,
  observations, reserved restart action, and continuation status.
- The store deliberately does not journal every append instruction. A process crash inside the append/cleanup boundary
  may lose or duplicate a small bounded shard set, and header tearing fails loudly rather than being inferred.
- A pending optimizer quantum records exact replay identity and completed-model prefix; restart refuses a changed
  snapshot. Checkpoint manifests are written last and one complete uncommitted successor may be adopted atomically.

### Decision rationale

- The current design favors bounded work, explicit ownership, typed manifests, and simple recovery over globally
  deterministic ingestion order or an elaborate write-ahead log.
- Quarantining one corrupt game prevents a data-quality incident from becoming a silent training stall, while the
  rolling alarm prevents quarantine from hiding systemic schema failure.
- Memory mapping and vectorized gather remove the old need for compaction solely to make thousands of tiny files
  trainable. Historical compaction remains relevant evidence about the bottleneck, not part of the active design.

### Pitfalls and unknowns

- The old queue-based materializer could enter a self-amplifying CPU wedge as inbox depth grew. Benchmarking only
  steady state would have missed the scaling failure.
- Approximate cross-worker append order slightly perturbs FIFO age. The project accepts this but should report actual
  age distribution.
- The bounded dirty-restart loss/duplication window has not been quantified from a fault-injection benchmark.
- The final report needs rejection counts, dispatcher backlog, worker throughput, staged-shard occupancy, append/flush
  time, prefetch wait, host-to-device time, and trainer starvation.
- Replay schema migration is offline-only. An incompatible store must never be rewritten under a live run.

### Sources

- [Current replay-pipeline design](../../architecture/replay-pipeline-rework.md)
- [Materialization failure analysis and replacement design](../../architecture/replay-materialization-rework.md)
- [Dispatcher](../../../py/src/replay/dispatch.py)
- [Materialization worker](../../../py/src/replay/materialization_worker.py)
- [Replay manager](../../../py/src/replay/manager.py)
- [Columnar store](../../../py/src/replay/store.py)
- [Mapped loader and prefetch](../../../py/src/replay/batch_loader.py)
- [Historical loader benchmark](../../benchmarks/replay-loader-20260724/README.md)
- [Production DDP benchmark](../../benchmarks/ddp-production-training-20260720/README.md)
- [Vectorized batch construction commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/dc6ab37e)
- [Mapped prefetch commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/f6af39bd)
- [Per-worker dispatch implementation](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/8d22010f)

## Publication-safe conclusions

- The retained data system combines random opening prefixes, archived difficult-state restarts, calibrated
  resignation, direct cut-position bootstraps, staged replay growth, policy-surprise sampling, unit loss weights, and
  reuse four. Most are components of one successful bundle; few have isolated online Elo estimates.
- The strongest causal result in this dossier is the cut-target comparison: searched root value beat the material
  heuristic on proper scoring rules in both measured cohorts.
- The strongest incident lesson is late-game poisoning: excluding deep policy rows while broadcasting one shallow cut
  value across an entire game can leave weight damage after the offending rows disappear from replay.
- Short stationary-replay screens are valuable for rejecting unstable optimizer, folding, and learning-rate choices,
  but they do not establish online strength.
- Replay engineering results establish throughput, durability, and correctness. They should be connected to strength
  only through measured admitted-data rate, optimizer cadence, and wall-clock learning.

## Evidence still required from the terminal archive

- Effective presentations per admitted row and per estimated unique state.
- Replay occupancy, age percentiles, duplicate estimates, surprise distribution, and sampling probability by age and
  game phase.
- Fresh rows and games per second, credit starvation, quanta per hour, trainer duty cycle, and actor overlap.
- Random-opening lengths, restart eligibility and fallback, archive occupancy, reservations, and eviction reasons.
- Resignation threshold trajectory, audit counts, confidence bounds, false non-losses, actual resignations, and saved
  plies/search.
- Natural/resignation/cut termination mix, cut-value calibration, remaining-length eligibility, and auxiliary
  eligibility rates.
- Materialization backlog, rejection counts, staged-shard occupancy, append/flush time, prefetch wait, and recovery
  incidents.
- Terminal-target entropy, root-value calibration, the effective root-blend schedule, and the independent search- and
  replay-discount settings.
- Candidate-local optimizer steps, multiplier-supplied wall time, loss-comparison EMAs, promotion decisions, and
  replay provenance around every model transition.
- End-to-end checkpoint publication time, actor refresh lag, and the age and volume of data produced by each public
  model.
- Recovery or explicit qualification of the short replay-reuse control curves whose current provenance is only Git
  history.
