# 7. Integrated chess recipe

The final recipe combines progressive network sizing, fully searched self-play, targeted restarts, and prioritized
replay with quantization-aware training for INT8 deployment. It ran on one node with eight NVIDIA GeForce
RTX 4070 SUPER GPUs and 80 logical CPUs. This chapter specifies how the retained methods operate together;
[Appendix D](appendix-d-reproducibility.md) provides the numerical settings, and the fully expanded
`chess-final-config.yaml` [10] is the reproduction entry point.

## Network and learner

The configured network ladder contains 12×128, 14×160, and 19×176 residual CNNs, where the notation gives residual
blocks and channels. All stages use capped scaled post-activation branches, global-pooling context in every
second block, a from-to policy head, and a compact WDL head. The next-policy and remaining-game-length auxiliary
heads contribute during training and are excluded from inference.

The primary objectives fit searched policies and WDL outcomes. Outcome targets use per-ply discounting and a
scheduled search-root-value blend. Capped games receive a searched-root-value target instead of a terminal
outcome and contribute no remaining-game-length loss. Appendix D specifies the weights and discount factors.

Training runs in 500-step blocks across eight GPUs, using Nesterov SGD and bfloat16 with a global batch of 2,048.
Quantization-aware training begins with the first updates. Batch normalization remains separate in the trainable
model; folding it into convolutions, recalibrating, exporting to ONNX, and refitting TensorRT all happen on a
deployment copy. This separates the optimizer's model from the compiled representation used by self-play.
Appendix D gives the learning-rate and calibration settings.

## Game supply and replay

Thirty-two actors run self-play, four per GPU, with 512 games interleaved in each process. Half the actors on
each GPU pause during a training quantum; the remainder continue searching to replenish replay. This is the
half-active scheduling policy evaluated in Chapter 5.

The search schedule increases the fixed budget from 300 to 800 visits per move; the reported checkpoint was
reached during the 600-visit stage. Adaptive allocation and learned stopping were not retained. Appendix D
specifies the search constants and native inference batch settings.

Games begin approximately equally often from up to eight random legal opening plies and from recent restart
states. The archive favours recent, undecided positions with promising unexplored alternatives; when no suitable
restart is available, the game uses a random opening. Game caps grow from 150 to 250 plies, and greedy move selection begins
later as training matures. Resignation is calibrated against no-resignation games with a 2.5% upper-bound target
on mistakenly resigning a draw or win; 20% of games are designated as such safety continuations at creation.

Replay capacity grows through ten stages from 600,000 to 20 million live rows. Each admitted position funds four
training presentations. A 30% uniform sampling component maintains broad coverage, while the remaining
probability prioritizes bounded policy surprise. Training credit is issued after completed-game positions have
been admitted to the persistent replay store.

## Sizing and promotion

When the active model's smoothed 64-search Elo gain falls below the stage threshold, its successor begins training
on the same replay. The thresholds are 15 Elo/hour for the small stage and 4 Elo/hour for the medium stage.
The candidate receives extra catch-up training, then takes over after scoring at least 0.48 in two consecutive
paired matches against the active model. Appendix D gives the catch-up schedule.

A separate capacity study grew the trained 14×160 network into 19×176 while preserving its function, then
recovered deployment fidelity through QAT. The limited continuation reached parity but did not establish a
higher plateau; the reported checkpoint is therefore the 14×160 model.

## Measuring progress during training

Training progress was evaluated every 20 minutes using 50 paired openings and a 64-search ladder against a
three-rung adaptive bracket of fixed-node Stockfish 13 opponents. A policy-only ladder used the same openings and
opponent framework. These monitors informed candidate timing; the final evaluation used the separate matches
reported in Chapters 2 and 8.
