# 7. Integrated chess recipe

The final chess recipe balances three needs. Smaller networks make early searched self-play affordable; the learner
later moves to a stronger network when progress slows. Games continue to produce searched positions through the
endgame, while restart states and replay sampling return informative positions to training. Quantization-aware
training lets the model that learns also serve the INT8 search system. These choices ran together on one node with
eight NVIDIA GeForce RTX 4070 SUPER GPUs and 80 logical CPUs. The fully expanded `chess-final-config.yaml` [10]
supports new runs; Appendix D records the secondary settings and distinguishes this living recipe from the frozen
result.

## Network and learner

The configured network ladder contains 12×128, 14×160, and 19×176 residual CNNs. Each has capped scaled
post-activation branches and global-pooling context in every second block. A from-to policy head predicts chess
moves, while a two-channel WDL head predicts outcomes. Training-only heads also predict the next searched policy
and remaining game length; deployment removes them.

Policy and value are the primary training targets. Terminal outcomes are discounted per ply, with a small
search-root-value blend introduced over training. If a game is cut short, the searched root value substitutes for
an unknown terminal result, and its unknown remaining-game-length label is censored. Appendix D gives the weights
and discount factors.

Training runs in 500-step quanta on eight ranks with a global batch of 2,048, using Nesterov SGD and bfloat16.
The trainable network uses quantization-aware training (QAT) from the beginning. Its parameters remain pre-fold:
folding, recalibration, ONNX export, and TensorRT refitting happen on a deployment copy. This protects the
optimizer's model while producing the INT8 artifact used for searched play. Appendix D gives optimizer,
calibration, and learning-rate settings.

## Game supply and replay

Thirty-two actors run self-play, four per GPU, with 512 games interleaved in each process. Half the actors on
each GPU pause during a training quantum; the rest continue searching. This overlap keeps new games arriving
without surrendering the whole node to either training or self-play. Chapter 5 measures the throughput tradeoff.

The fixed search budget rises from 300 to 800 visits per move. Fixed visits remained in the recipe after adaptive
allocation and learned stopping failed to improve wall-clock learning. Appendix D gives the search constants and
native inference batch settings.

Games begin approximately equally often from up to eight random legal opening plies and from recent restart
states. Restart candidates are screened for value, remaining length, age, and branchable visit mass; an unavailable
restart falls back to a random opening. Game caps grow from 150 to 250 plies, and greedy move selection begins
later as training matures. Resignation is calibrated against no-resignation games so that the upper bound on a
false nonloss stays below 2.5%; 20% of games are designated as such safety continuations at creation.

Replay capacity grows through ten stages from 600,000 to 20 million live rows. Each admitted position funds four
training presentations. Sampling keeps 30% uniform probability while otherwise prioritizing bounded policy
surprise, so difficult positions are revisited without excluding the wider game distribution. Completed games
become durable rows before training credit is committed.

## Sizing and promotion

The smoothed 64-search ladder signals when an active model reaches a stage-specific plateau. Its successor then
trains on a replay snapshot with a separate catch-up learning-rate clock. Candidate-versus-active games, rather
than training loss, decide promotion: extra candidate presentations had made earlier loss comparisons misleading
and had admitted a much weaker model. Appendix D gives the catch-up ratio and match gate.

A separate capacity study grew the trained 14×160 network into 19×176 while preserving its function, then
recovered deployment fidelity through QAT. The larger continuation reached parity without establishing a stronger
plateau, so the selected checkpoint remains the 14×160 model. That continuation was too short to determine why
further growth did not help.

## Measuring progress during training

Every 20 minutes, 50 paired openings supported a 64-search ladder against a three-rung adaptive bracket of
fixed-node Stockfish 13 opponents. A policy-only ladder used the same openings and opponent framework. These
measurements guided progress and candidate timing; the terminal matches reported in Chapters 2 and 8 are a
separate evaluation.
