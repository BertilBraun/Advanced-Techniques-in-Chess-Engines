### Experiment status

Every method has three checkboxes:

* `implemented`: the production path can run the method as described;
* `experimenting`: a dedicated one-variable screening run is currently configured or active;
* `validated`: an informative completed experiment has established that the method improves the selected baseline.

Leave `validated` unchecked until a four-hour run reaches the difficult part of training and the result is accepted.
If a completed experiment does not improve the baseline, replace `validated [ ]` with `validated [✗]` and retain the
result rather than removing the method from the history.

Current screening policy:

* Compare one method at a time against the same baseline before combining accepted improvements.
* Use four-hour runs for the current 7x7 Go screen; the earlier two-hour runs are diagnostic evidence only.
* With four concurrent experiments, 24 hours of continuous work requires 24 four-hour runs.
* Use spare capacity for new single-variable methods and confirmation seeds of promising results.
* Defer dynamic per-position budgets, Gumbel search, and progressive model scaling for the current screen.

### Highest-priority training-efficiency additions

* **Progressive simulation budget** - implemented [x] experimenting [x] validated [ ]

  * Start with very low MCTS budgets.
  * Increase simulations as the network improves.
  * Test fixed schedules first, then performance-triggered schedules.
  * Strong candidate because early search over a weak network is often wasteful.

* **KataGo-style mixed fast/full searches** - implemented [x] experimenting [x] validated [ ]

  * Most moves use cheap, exploitative search.
  * A minority use expensive full search.
  * Train policy primarily from full-search positions.
  * Goal: more completed games and more independent value targets per hour.

* **Fast natural-outcome continuation** - implemented [x] experimenting [ ] validated [ ]

  * Optionally force fast searches after a configured ply while retaining the independently scheduled emergency cap.
  * Fast continuation uses the existing greedy late-game selection, noise-free fast search, and tree reuse.
  * Fast observations stay out of replay but carry the natural terminal outcome back to earlier full-search samples.
  * A restart beyond the cutoff retains one full search at its reserved branch root before entering fast continuation.
  * The chess template starts continuation at ply 250 while its game-specific material-adjudication cap grows from
    200 to 600 plies.

* **Adaptive search termination** - implemented [ ] experimenting [ ] validated [ ]

  * Stop before the nominal simulation cap when one move is clearly dominant.
  * Require a minimum search count, visit dominance, Q-margin, and possibly an unrecoverable visit lead.
  * Calibrate offline against full-search traces before training with it.

* **Progressive model scaling** - implemented [ ] experimenting [ ] validated [ ]

  * Small network early, larger network later.
  * Train the larger model on the same replay data before promotion.
  * Compare fixed-time promotion, loss crossover, and Elo crossover.
  * KataGo used progressive network growth successfully.
  * Deferred because model transition overhead is not justified in a four-hour screen.

* **Optimize the self-play/training ratio** - implemented [x] experimenting [x] validated [ ]

  * Vary optimizer steps per generated position.
  * Track sample reuse explicitly.
  * Too little training wastes data; too much causes overfitting to stale replay.

* **Optimize replay-window size** - implemented [x] experimenting [ ] validated [ ]

  * Compare small recent buffers against larger historical windows.
  * Potentially grow the replay window over training.
  * Track replay age, sample reuse, and policy staleness.

* **Optimize model publication cadence** - implemented [x] experimenting [ ] validated [✗]

  * Publish a new self-play model every fixed number of optimizer steps.
  * Compare frequent publication against larger, less frequent updates.
  * Very frequent publication may create overhead and unstable moving targets; infrequent publication wastes improved models.
  * A 100-optimizer-step cadence underperformed because publication overhead exceeded the freshness benefit.

---

### Search improvements

* **First-play urgency ablation** - implemented [x] experimenting [x] validated [ ]

  * Zero initialization.
  * Parent-value initialization.
  * Reduced parent value with an explicit fixed reduction.
  * Mean visited-child Q with pessimistic virtual evidence.
  * Especially important for low simulation budgets.
  * The first 7x7 R14 implementation accidentally made parent-value FPU the meaning of a zero reduction, so every
    second-round run changed FPU relative to the true-zero R13 baseline. Do not use those runs as clean
    single-variable evidence.
  * The current typed modes restore zero as the baseline and expose parent-value and reduced-parent-value FPU as
    separate treatments. Do not call the fixed-reduction treatment KataGo-style without matching its full formula.

* **Forced playouts with policy-target pruning** - implemented [x] experimenting [x] validated [ ]

  * Force exploration of root candidates.
  * Remove visits caused only by exploration from the supervised policy target.
  * Separates search exploration from policy supervision.
  * The existing uniform minimum-root-visit preprocessing is scaffolding, not the intended prior-scaled method.

* **Gumbel search** - implemented [ ] experimenting [ ] validated [ ]

  * Most relevant at very low simulation budgets.
  * Test separately:

    * Gumbel candidate sampling;
    * sequential halving;
    * completed-Q policy targets;
    * transformed-Q interior selection.
  * Do not treat it as one indivisible block.
  * Deferred while full searches remain in the 64-512 simulation range.

* **Deterministic sequential halving** - implemented [ ] experimenting [ ] validated [ ]

  * Use top-(k) prior actions without Gumbel noise.
  * Useful for cheap exploitative moves where the goal is simply to find the best move quickly.

* **Dynamic root candidate count** - implemented [ ] experimenting [ ] validated [ ]

  * Choose (k) from policy entropy or effective support.
  * Small (k) for confident positions, larger (k) for uncertain positions.

* **Dynamic simulation budget per position** - implemented [ ] experimenting [ ] validated [ ]

  * Spend more search on high-entropy or close-value positions.
  * Spend less on obvious positions.
  * More principled than a globally fixed simulation cap.
  * Deferred because calibrated adaptive termination captures the immediate benefit without a separate difficulty
    estimator.

* **Tree reuse across moves** - implemented [x] experimenting [x] validated [ ]

  * Reuse the chosen child subtree after playing a move.
  * Straightforward inference savings.
  * Must be handled carefully when root noise or policy targets differ.

* **Transposition-aware graph search** - implemented [ ] experimenting [ ] validated [ ]

  * Share evaluations and statistics between identical states reached by different move orders.
  * Particularly relevant for chess.
  * Requires correct treatment of repetition and history-dependent legality.

* **Search batching improvements** - implemented [x] experimenting [ ] validated [x]

  * Multiple simultaneous games per worker.
  * Batched leaf evaluation.
  * Virtual loss or equivalent collision handling.
  * Often more important practically than a small algorithmic improvement.

* **Asynchronous self-play** - implemented [ ] experimenting [ ] validated [ ]

  * Avoid global barriers between self-play and training.
  * Keep GPUs saturated.
  * Measure whether increased policy staleness outweighs utilization gains.
  * The current runtime overlaps selected self-play workers with optimizer work, but does not implement fully
    asynchronous model publication and training.

---

### Data-generation improvements

* **Go-Exploit-style restart states** - implemented [x] experimenting [x] validated [ ]

  * Start some trajectories from recent archived positions.
  * Produces shorter games, deeper-state coverage, and more independent terminal outcomes.
  * Retain a substantial probability of starting from the true initial state.
  * Prefer branchable archived positions with at least two plausible actions and enough remaining game length.
  * Track already sampled actions per archived position so a restart explores an untried continuation.

* **Branching from selected positions** - implemented [ ] experimenting [ ] validated [ ]

  * Generate multiple continuations from strategically interesting states.
  * Useful when search uncertainty is high or top actions are close.

* **Prioritize difficult states** - implemented [ ] experimenting [ ] validated [ ]

  * Sample restart or replay states using:

    * value error;
    * search disagreement;
    * policy entropy;
    * novelty;
    * large policy updates;
    * high estimated regret.

* **Reanalysis** - implemented [ ] experimenting [ ] validated [ ]

  * Re-search old replay states with a newer network.
  * Refresh stale policy targets.
  * Compare reanalysis compute against generating new games.

* **Resignation** - implemented [ ] experimenting [ ] validated [ ]

  * Introduce only after value calibration is adequate.
  * Use conservative thresholds and retain a fraction of non-resigning games.
  * Saves large amounts of late-game search.

* **Draw and repetition handling** - implemented [x] experimenting [ ] validated [ ]

  * Especially important for chess.
  * Ensure repeated-state information is present in the state representation.
  * Incorrect handling can contaminate value targets.

* **Opening diversity** - implemented [x] experimenting [x] validated [ ]

  * Dirichlet noise.
  * Temperature-based sampling.
  * Randomized opening prefixes.
  * Small archive-based restarts.
  * Avoid excessive exploration that weakens every game.

* **Position filtering** - implemented [ ] experimenting [ ] validated [ ]

  * Downweight or remove duplicate, trivial, forced, or low-information positions.
  * Be careful not to distort value training.

---

### Generic auxiliary targets

* **Opponent's next policy** - implemented [x] experimenting [x] validated [ ]

  * Cheap because the next search target already exists.
  * Generic across sequential games.
  * KataGo found a modest but clear benefit.

* **Remaining game length** - implemented [x] experimenting [x] validated [ ]

  * Predict moves or plies until termination.
  * Provides phase information.
  * Cheap exact labels.

* **Future own action** - implemented [ ] experimenting [ ] validated [ ]

  * Predict the action one or several plies ahead.
  * Generic temporal representation learning.

* **Search value distribution** - implemented [ ] experimenting [ ] validated [ ]

  * Predict a distribution over returns rather than only the expectation.
  * Useful for uncertainty and calibration.

* **Root Q-values** - implemented [ ] experimenting [ ] validated [ ]

  * Predict search-improved Q-values for selected actions.
  * Can make the policy target more informative than visit counts alone.

* **Outcome type** - implemented [x] experimenting [ ] validated [ ]

  * Win, loss, draw.
  * For chess, optionally distinguish mate, repetition, fifty-move rule, and insufficient material only as a secondary ablation.

* **Value at multiple horizons** - implemented [x] experimenting [x] validated [ ]

  * Terminal outcome.
  * Bootstrapped short-horizon value.
  * Search value.
  * Requires careful loss weighting to avoid self-reinforcing bias.

* **Uncertainty or variance head** - implemented [ ] experimenting [ ] validated [ ]

  * Predict return variance or search instability.
  * Could drive adaptive search budgets.

---

### Game-specific auxiliary targets worth testing separately

* **Chess material balance** - implemented [ ] experimenting [ ] validated [ ]

  * Generic-looking but still chess-specific.
  * Cheap, exact, and likely useful early.

* **Piece survival or capture targets** - implemented [ ] experimenting [ ] validated [ ]

  * Predict which pieces remain after a future horizon or at termination.

* **King safety** - implemented [ ] experimenting [ ] validated [ ]

  * Strongly game-specific and harder to define cleanly.
  * Probably not suitable for the main “minimal knowledge” system.

* **Move-count-to-mate or terminal distance** - implemented [ ] experimenting [ ] validated [ ]

  * Useful in tactical/endgame positions.
  * Sparse and potentially hard to calibrate.

* **Attack or control maps** - implemented [ ] experimenting [ ] validated [ ]

  * Dense supervision.
  * Encodes substantial chess structure.

* **Legal-move prediction** - implemented [ ] experimenting [ ] validated [ ]

  * Usually redundant because the legal mask is known.
  * Could regularize representation, but likely low priority.

For the main paper, keep these out of the primary method. Use them only as an upper-bound ablation on how much domain-specific supervision helps.

---

### Architecture changes

* **Global-context conditioning** - implemented [x] experimenting [ ] validated [ ]

  * Global pooling or squeeze-excitation blocks.
  * Useful because local move preferences depend on global state.
  * KataGo found a large efficiency benefit.
  * The current baseline uses squeeze-excitation every second residual block; dedicated KataGo-style global pooling
    remains a separate possible ablation.

* **Residual-network width/depth schedules** - implemented [ ] experimenting [ ] validated [ ]

  * Test width versus depth separately.
  * Smaller early models may benefit more from width than depth, or vice versa.

* **Policy/value trunk sharing** - implemented [x] experimenting [ ] validated [ ]

  * Fully shared trunk.
  * Partially separated late blocks.
  * Separate value capacity can help if value learning is the bottleneck.

* **Transformer or hybrid trunk** - implemented [ ] experimenting [ ] validated [ ]

  * Potentially better global interaction modeling.
  * More difficult to make compute-efficient at small board sizes.
  * Lower priority than improving the existing residual network.

* **Efficient convolutions** - implemented [x] experimenting [ ] validated [x]

  * Fused kernels.
  * Channels-last memory format.
  * Mixed precision.
  * Tensor-core-aligned channel counts.
  * The current inference artifact fuses convolution, batch normalization, and activation where applicable; not all
    listed kernel and memory-format variants are implemented.

* **Quantized self-play inference** - implemented [x] experimenting [ ] validated [x]

  * FP16/BF16 first.
  * INT8 only if accuracy and batching remain stable.
  * Training can stay higher precision.
  * CUDA self-play inference currently uses BF16; INT8 remains unimplemented.

* **Compile and fuse inference** - implemented [x] experimenting [ ] validated [x]

  * CUDA graphs.
  * `torch.compile`.
  * Static shapes where possible.
  * Eliminate Python overhead in self-play loops.
  * Model fusion and native static-shape inference are implemented; CUDA graphs and `torch.compile` are not.

---

### Training improvements

* **Mixed precision** - implemented [ ] experimenting [ ] validated [ ]

  * BF16 where supported.
  * FP16 with proper loss scaling otherwise.
  * Self-play inference uses BF16, but optimizer training remains FP32.

* **Optimizer comparison** - implemented [x] experimenting [ ] validated [ ]

  * SGD with momentum.
  * AdamW.
  * Lion or other alternatives only if justified.
  * Compare strength per wall-clock hour, not loss per step.

* **Learning-rate schedule** - implemented [x] experimenting [x] validated [ ]

  * Warm-up.
  * Cosine decay.
  * Piecewise drops.
  * Scale with effective batch and replay freshness.

* **Weight averaging** - implemented [ ] experimenting [ ] validated [ ]

  * EMA or stochastic weight averaging.
  * Evaluate whether averaged weights improve self-play stability and strength.

* **Gradient accumulation** - implemented [ ] experimenting [ ] validated [ ]

  * Useful only when hardware memory limits batch size.
  * It may hurt update frequency and freshness.

* **Loss balancing** - implemented [ ] experimenting [ ] validated [ ]

  * Dynamic or normalized weighting across policy, value, and auxiliary heads.
  * Track gradient norms and interference.

* **Prioritized replay** - implemented [ ] experimenting [ ] validated [ ]

  * Priority by value error, policy loss, or search disagreement.
  * Correct for sampling bias if needed.
  * Compare against simple uniform recent replay.

* **Recency weighting** - implemented [ ] experimenting [ ] validated [ ]

  * Soft weighting by age instead of a hard replay cutoff.

* **Data deduplication** - implemented [ ] experimenting [ ] validated [ ]

  * Prevent repeated opening positions from dominating training.

* **Target freshness tracking** - implemented [ ] experimenting [ ] validated [ ]

  * Record the network version that generated each policy target.
  * Use this directly in replay sampling or weighting.
  * Replay already records source generation and timestamp, but no sampler or loss currently uses them.

---

### Augmentation

* **Board symmetries** - implemented [x] experimenting [ ] validated [x]

  * Rotations and reflections when exactly valid.
  * For chess, only horizontal reflection is generally straightforward; colour/perspective canonicalization is more important.

* **Player-perspective canonicalization** - implemented [x] experimenting [ ] validated [x]

  * Always represent the side to move consistently.
  * Reduces the effective state space.

* **Colour swapping** - implemented [ ] experimenting [ ] validated [ ]

  * Valid only with correctly transformed castling, en passant, and move history.

* **Randomized rule variants** - implemented [ ] experimenting [ ] validated [ ]

  * Probably not useful for the primary chess engine.
  * More relevant for generalization studies.

* **Policy-target smoothing** - implemented [ ] experimenting [ ] validated [ ]

  * Small label smoothing or visit-count temperature.
  * Must not erase meaningful search concentration.

* **Value-target smoothing** - implemented [ ] experimenting [ ] validated [ ]

  * Potentially useful for noisy self-play, but binary terminal outcomes are exact.
  * Lower priority.

---

### Evaluation and experiment-control features

* **Same hardware and wall-clock duration** - implemented [x] experimenting [x] validated [ ]

  * Primary fairness criterion.

* **Checkpoint evaluation at fixed elapsed times** - implemented [x] experimenting [x] validated [ ]

  * Example: 1, 2, 4, 8, 12, 24 hours.

* **Common evaluation search** - implemented [x] experimenting [x] validated [ ]

  * Isolates network quality.

* **Native-search evaluation** - implemented [x] experimenting [x] validated [ ]

  * Measures actual engine strength under its intended deployment setup.

* **Learning-curve AUC** - implemented [ ] experimenting [ ] validated [ ]

  * Primary training-efficiency metric.

* **Final Elo at fixed time** - implemented [ ] experimenting [ ] validated [ ]

  * Secondary primary metric.

* **Multiple seeds on small Go** - implemented [x] experimenting [ ] validated [ ]

  * At least 3 for screening.
  * 5–10 for claims.

* **Matched initialization** - implemented [x] experimenting [x] validated [ ]

  * Use the same initial weights and random seeds where possible.

* **Full utilization logging** - implemented [x] experimenting [x] validated [x]

  * GPU utilization.
  * Batch size.
  * inference latency.
  * games per hour.
  * positions per hour.
  * simulations per second.
  * optimizer steps per hour.
  * replay age.
  * idle time.

---

### Sensible implementation order

* Stable standard AlphaZero baseline.
* Progressive simulation budget.
* Mixed fast/full search.
* KataGo-style reduced-parent-value FPU.
* Go-Exploit-style restart states with untried-action tracking.
* Remaining-game-length auxiliary head.
* Forced playouts with policy-target pruning.
* Conservative adaptive search termination after offline trace calibration.
* Replay ratio and publication cadence.
* Dedicated global-pooling ablation beyond the existing squeeze-excitation blocks.
* Other generic auxiliary heads.
* Reanalysis only if replay staleness is measured as a bottleneck.
* Combined optimized system.
* Final chess run under the approximately $50 rental budget.

Deferred from the four-hour 7x7 Go screen:

* dynamic simulation budgets, because adaptive termination is the simpler evidence-driven mechanism;
* Gumbel and sequential halving, because full-search targets currently use 64-512 simulations;
* progressive model scaling, because shape transition and promotion overhead dominate a four-hour experiment;
* transposition-aware graph search, because its primary value is in the later chess stage.

### Current screening queue

The first four runs are active: baseline, learning-rate decay, constant learning rate, and mixed search with 25% full
searches. The remaining six pre-existing single-variable screens should still run. Four additional screens are now
configured for reduced-parent FPU, restart states, remaining game length, and forced playout pruning.

Review the first completed results before choosing another variable or allocating confirmation seeds. Adaptive search
termination and the broader 24-hour queue decision are deferred to `TOMORROW.md`.

### Likely final system

* Progressive simulation schedule.
* Mixed fast/full searches.
* Conservative adaptive early stopping.
* Strong FPU initialization.
* Forced root exploration with pruned policy targets.
* Archive-based restart states with controlled branch coverage.
* Growing recent replay window.
* Tuned optimizer-to-data ratio.
* Frequent but not excessive model publication.
* Global-context architecture.
* Symmetry/perspective augmentation.
* Opponent-policy and game-length auxiliary heads.
* High-throughput batched mixed-precision inference.
* Tree reuse and transposition handling for chess.
* Optional reanalysis only if it beats fresh self-play per wall-clock hour.
