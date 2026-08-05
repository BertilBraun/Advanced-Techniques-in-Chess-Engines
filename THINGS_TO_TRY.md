### Highest-priority training-efficiency additions

* **Progressive simulation budget**

  * Start with very low MCTS budgets.
  * Increase simulations as the network improves.
  * Test fixed schedules first, then performance-triggered schedules.
  * Strong candidate because early search over a weak network is often wasteful. 

* **KataGo-style mixed fast/full searches**

  * Most moves use cheap, exploitative search.
  * A minority use expensive full search.
  * Train policy primarily from full-search positions.
  * Goal: more completed games and more independent value targets per hour. 

* **Adaptive search termination**

  * Stop before the nominal simulation cap when one move is clearly dominant.
  * Require a minimum search count, visit dominance, Q-margin, and possibly an unrecoverable visit lead.
  * Calibrate offline against full-search traces before training with it.

* **Progressive model scaling**

  * Small network early, larger network later.
  * Train the larger model on the same replay data before promotion.
  * Compare fixed-time promotion, loss crossover, and Elo crossover.
  * KataGo used progressive network growth successfully. 

* **Optimize the self-play/training ratio**

  * Vary optimizer steps per generated position.
  * Track sample reuse explicitly.
  * Too little training wastes data; too much causes overfitting to stale replay.

* **Optimize replay-window size**

  * Compare small recent buffers against larger historical windows.
  * Potentially grow the replay window over training.
  * Track replay age, sample reuse, and policy staleness.

* **Optimize model publication cadence**

  * Publish a new self-play model every fixed number of optimizer steps.
  * Compare frequent publication against larger, less frequent updates.
  * Very frequent publication may create overhead and unstable moving targets; infrequent publication wastes improved models.

---

### Search improvements

* **First-play urgency ablation**

  * Zero initialization.
  * Parent-value initialization.
  * KataGo-style reduced parent value.
  * Mean visited-child Q with pessimistic virtual evidence.
  * Especially important for low simulation budgets. 

* **Forced playouts with policy-target pruning**

  * Force exploration of root candidates.
  * Remove visits caused only by exploration from the supervised policy target.
  * Separates search exploration from policy supervision. 

* **Gumbel search**

  * Most relevant at very low simulation budgets.
  * Test separately:

    * Gumbel candidate sampling;
    * sequential halving;
    * completed-Q policy targets;
    * transformed-Q interior selection.
  * Do not treat it as one indivisible block.

* **Deterministic sequential halving**

  * Use top-(k) prior actions without Gumbel noise.
  * Useful for cheap exploitative moves where the goal is simply to find the best move quickly.

* **Dynamic root candidate count**

  * Choose (k) from policy entropy or effective support.
  * Small (k) for confident positions, larger (k) for uncertain positions.

* **Dynamic simulation budget per position**

  * Spend more search on high-entropy or close-value positions.
  * Spend less on obvious positions.
  * More principled than a globally fixed simulation cap.

* **Tree reuse across moves**

  * Reuse the chosen child subtree after playing a move.
  * Straightforward inference savings.
  * Must be handled carefully when root noise or policy targets differ.

* **Transposition-aware graph search**

  * Share evaluations and statistics between identical states reached by different move orders.
  * Particularly relevant for chess.
  * Requires correct treatment of repetition and history-dependent legality.

* **Search batching improvements**

  * Multiple simultaneous games per worker.
  * Batched leaf evaluation.
  * Virtual loss or equivalent collision handling.
  * Often more important practically than a small algorithmic improvement.

* **Asynchronous self-play**

  * Avoid global barriers between self-play and training.
  * Keep GPUs saturated.
  * Measure whether increased policy staleness outweighs utilization gains.

---

### Data-generation improvements

* **Go-Exploit-style restart states**

  * Start some trajectories from recent archived positions.
  * Produces shorter games, deeper-state coverage, and more independent terminal outcomes.
  * Retain a substantial probability of starting from the true initial state. 

* **Branching from selected positions**

  * Generate multiple continuations from strategically interesting states.
  * Useful when search uncertainty is high or top actions are close.

* **Prioritize difficult states**

  * Sample restart or replay states using:

    * value error;
    * search disagreement;
    * policy entropy;
    * novelty;
    * large policy updates;
    * high estimated regret.

* **Reanalysis**

  * Re-search old replay states with a newer network.
  * Refresh stale policy targets.
  * Compare reanalysis compute against generating new games.

* **Resignation**

  * Introduce only after value calibration is adequate.
  * Use conservative thresholds and retain a fraction of non-resigning games.
  * Saves large amounts of late-game search.

* **Draw and repetition handling**

  * Especially important for chess.
  * Ensure repeated-state information is present in the state representation.
  * Incorrect handling can contaminate value targets.

* **Opening diversity**

  * Dirichlet noise.
  * Temperature-based sampling.
  * Randomized opening prefixes.
  * Small archive-based restarts.
  * Avoid excessive exploration that weakens every game.

* **Position filtering**

  * Downweight or remove duplicate, trivial, forced, or low-information positions.
  * Be careful not to distort value training.

---

### Generic auxiliary targets

* **Opponent’s next policy**

  * Cheap because the next search target already exists.
  * Generic across sequential games.
  * KataGo found a modest but clear benefit. 

* **Remaining game length**

  * Predict moves or plies until termination.
  * Provides phase information.
  * Cheap exact labels.

* **Future own action**

  * Predict the action one or several plies ahead.
  * Generic temporal representation learning.

* **Search value distribution**

  * Predict a distribution over returns rather than only the expectation.
  * Useful for uncertainty and calibration.

* **Root Q-values**

  * Predict search-improved Q-values for selected actions.
  * Can make the policy target more informative than visit counts alone.

* **Outcome type**

  * Win, loss, draw.
  * For chess, optionally distinguish mate, repetition, fifty-move rule, and insufficient material only as a secondary ablation.

* **Value at multiple horizons**

  * Terminal outcome.
  * Bootstrapped short-horizon value.
  * Search value.
  * Requires careful loss weighting to avoid self-reinforcing bias.

* **Uncertainty or variance head**

  * Predict return variance or search instability.
  * Could drive adaptive search budgets.

---

### Game-specific auxiliary targets worth testing separately

* **Chess material balance**

  * Generic-looking but still chess-specific.
  * Cheap, exact, and likely useful early.

* **Piece survival or capture targets**

  * Predict which pieces remain after a future horizon or at termination.

* **King safety**

  * Strongly game-specific and harder to define cleanly.
  * Probably not suitable for the main “minimal knowledge” system.

* **Move-count-to-mate or terminal distance**

  * Useful in tactical/endgame positions.
  * Sparse and potentially hard to calibrate.

* **Attack or control maps**

  * Dense supervision.
  * Encodes substantial chess structure.

* **Legal-move prediction**

  * Usually redundant because the legal mask is known.
  * Could regularize representation, but likely low priority.

For the main paper, keep these out of the primary method. Use them only as an upper-bound ablation on how much domain-specific supervision helps.

---

### Architecture changes

* **Global-context conditioning**

  * Global pooling or squeeze-excitation blocks.
  * Useful because local move preferences depend on global state.
  * KataGo found a large efficiency benefit. 

* **Residual-network width/depth schedules**

  * Test width versus depth separately.
  * Smaller early models may benefit more from width than depth, or vice versa.

* **Policy/value trunk sharing**

  * Fully shared trunk.
  * Partially separated late blocks.
  * Separate value capacity can help if value learning is the bottleneck.

* **Transformer or hybrid trunk**

  * Potentially better global interaction modeling.
  * More difficult to make compute-efficient at small board sizes.
  * Lower priority than improving the existing residual network.

* **Efficient convolutions**

  * Fused kernels.
  * Channels-last memory format.
  * Mixed precision.
  * Tensor-core-aligned channel counts.

* **Quantized self-play inference**

  * FP16/BF16 first.
  * INT8 only if accuracy and batching remain stable.
  * Training can stay higher precision.

* **Compile and fuse inference**

  * CUDA graphs.
  * `torch.compile`.
  * Static shapes where possible.
  * Eliminate Python overhead in self-play loops.

---

### Training improvements

* **Mixed precision**

  * BF16 where supported.
  * FP16 with proper loss scaling otherwise.

* **Optimizer comparison**

  * SGD with momentum.
  * AdamW.
  * Lion or other alternatives only if justified.
  * Compare strength per wall-clock hour, not loss per step.

* **Learning-rate schedule**

  * Warm-up.
  * Cosine decay.
  * Piecewise drops.
  * Scale with effective batch and replay freshness.

* **Weight averaging**

  * EMA or stochastic weight averaging.
  * Evaluate whether averaged weights improve self-play stability and strength.

* **Gradient accumulation**

  * Useful only when hardware memory limits batch size.
  * It may hurt update frequency and freshness.

* **Loss balancing**

  * Dynamic or normalized weighting across policy, value, and auxiliary heads.
  * Track gradient norms and interference.

* **Prioritized replay**

  * Priority by value error, policy loss, or search disagreement.
  * Correct for sampling bias if needed.
  * Compare against simple uniform recent replay.

* **Recency weighting**

  * Soft weighting by age instead of a hard replay cutoff.

* **Data deduplication**

  * Prevent repeated opening positions from dominating training.

* **Target freshness tracking**

  * Record the network version that generated each policy target.
  * Use this directly in replay sampling or weighting.

---

### Augmentation

* **Board symmetries**

  * Rotations and reflections when exactly valid.
  * For chess, only horizontal reflection is generally straightforward; colour/perspective canonicalization is more important.

* **Player-perspective canonicalization**

  * Always represent the side to move consistently.
  * Reduces the effective state space.

* **Colour swapping**

  * Valid only with correctly transformed castling, en passant, and move history.

* **Randomized rule variants**

  * Probably not useful for the primary chess engine.
  * More relevant for generalization studies.

* **Policy-target smoothing**

  * Small label smoothing or visit-count temperature.
  * Must not erase meaningful search concentration.

* **Value-target smoothing**

  * Potentially useful for noisy self-play, but binary terminal outcomes are exact.
  * Lower priority.

---

### Evaluation and experiment-control features

* **Same hardware and wall-clock duration**

  * Primary fairness criterion.

* **Checkpoint evaluation at fixed elapsed times**

  * Example: 1, 2, 4, 8, 12, 24 hours.

* **Common evaluation search**

  * Isolates network quality.

* **Native-search evaluation**

  * Measures actual engine strength under its intended deployment setup.

* **Learning-curve AUC**

  * Primary training-efficiency metric.

* **Final Elo at fixed time**

  * Secondary primary metric.

* **Multiple seeds on small Go**

  * At least 3 for screening.
  * 5–10 for claims.

* **Matched initialization**

  * Use the same initial weights and random seeds where possible.

* **Full utilization logging**

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
* Adaptive search termination.
* FPU variants.
* Replay ratio and publication cadence.
* Progressive model scaling.
* Global pooling.
* Generic auxiliary heads.
* Reanalysis or restart-state self-play.
* Gumbel or sequential halving.
* Combined optimized system.
* Final chess run under the approximately $50 rental budget.

### Likely final system

* Small-to-large progressive model schedule.
* Progressive simulation schedule.
* Mixed fast/full searches.
* Conservative adaptive early stopping.
* Strong FPU initialization.
* Growing recent replay window.
* Tuned optimizer-to-data ratio.
* Frequent but not excessive model publication.
* Global-context architecture.
* Symmetry/perspective augmentation.
* Opponent-policy and game-length auxiliary heads.
* High-throughput batched mixed-precision inference.
* Tree reuse and transposition handling for chess.
* Optional reanalysis only if it beats fresh self-play per wall-clock hour.
