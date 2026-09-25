# 7. Integrated chess recipe

## Living recipe and frozen result

The fully expanded `chess-final-config.yaml` [10] is the entry point for reproducing or extending the chess system.
It brings the network ladder, trainer, replay, self-play, deployment, and online-evaluation settings into one recipe.
The file may evolve with the project; the experiment reported here is tied instead to the selected checkpoint and
frozen result record described in Chapter 8 and Appendices B and D.

Training used a node with eight NVIDIA GeForce RTX 4070 SUPER GPUs and 80 logical CPUs. The throughput results in
Chapter 5 were measured on stated workloads on this class of node.

## Model, objective, and optimization

The configured ladder contains 12×128, 14×160, and 19×176 residual CNNs. Every stage uses capped scaled
post-activation branches, global-pooling context in every second block, a key-size-128 chess from-to policy head, and
a two-channel WDL value head with a 48-unit hidden layer. Training-only heads predict the next searched policy at
weight 0.15 and remaining game length at weight 0.1; deployment removes both.

The primary policy and value losses each have weight 1.0. Terminal outcome targets are discounted by 0.998 per ply.
A search-root-value blend rises from zero to 0.1 over its configured schedule, while search backup uses a separate
0.99 per-ply discount. Cut games use the searched root value and censor unknown remaining-length labels.

Each training quantum contains 500 optimizer steps. Eight ranks each process 256 positions for a global batch of
2,048. The optimizer is Nesterov SGD with momentum 0.9 and weight decay 0.0001; gradients are clipped to norm 1.0.
The learning rate warms from zero to 0.1 over the first 1,000 optimizer steps and then follows the configured linear
schedule to 0.01. Training uses bfloat16 and persistent trainer processes; `torch.compile` is disabled.

INT8 quantization-aware training is active from the beginning. Calibration uses 516 real evaluation positions and is
refreshed at every publication boundary. The trainable model remains pre-fold; deployment folding, recalibration,
ONNX export, and TensorRT refitting operate on a copy. This keeps serving conversion from changing the optimizer's
parameterization.

## Self-play, replay, and curriculum

Self-play runs 32 actor processes, four assigned to each GPU, with 512 interleaved games per process. During a
training quantum, half of the actor processes on every GPU pause while the others continue producing games. Native
inference uses batch 320 with two outstanding batches per worker.

The fixed search budget grows from 300 to 800 visits per move. Search uses exploration constant 1.5, reduced-parent
FPU with reduction 0.2, forced playout coefficient 1.5, Dirichlet epsilon 0.25, and alpha 0.3. Fixed visits were
retained after adaptive allocation and learned stopping failed to improve wall-clock strength.

The opening draw aims to split games evenly between up to eight random legal plies and recent restart states. Restart
candidates are filtered by value, remaining length, age, and branchable visit mass, with a 30% uniform component in
selection. If no restart candidate qualifies, that game falls back to a random opening. Game caps grow
from 150 to 250 plies, and greedy move selection begins later as training matures. Calibrated resignation begins only
after sufficient evidence, constrains the false-nonloss upper bound to 2.5%, and designates 20% of games at creation
as no-resignation continuations for ongoing safety measurement.

Replay capacity grows through ten stages from 600,000 to 20 million live rows. Each admitted position funds four
training presentations. Sampling reserves 30% uniform probability and otherwise prioritizes bounded policy surprise.
Eight materializers convert completed games into the fixed-layout memory-mapped store; credit is committed only
against durable admitted data.

## Progressive sizing

The smoothed 64-search ladder triggers a successor when the active model reaches a stage-specific plateau. The
configuration [10] gives the thresholds and window. The successor then trains on a captured replay snapshot for an
average of 1.5 optimizer quanta per active-model quantum, with its own catch-up learning-rate clock.

Promotion is decided by candidate-versus-active matches, not training loss. The configured gate requires a score of
at least 0.48 in two consecutive paired evaluations. This distinction matters because extra candidate presentations
made the earlier loss comparison systematically favorable to the candidate and admitted a much weaker model.

A separate capacity study grew the trained 14×160 network into 19×176 while preserving its function, then recovered
deployment fidelity through QAT. The larger continuation reached parity but did not establish a stronger plateau,
so the selected checkpoint remains the 14×160 model. The continuation was too short to determine why further growth
did not help.

## Training-time and terminal evaluation

During training, evaluation ran every 20 minutes on 50 paired openings. The searched ladder used 64 searches per move
against a three-rung adaptive bracket of fixed-node Stockfish 13 opponents; a policy-only ladder used the same opening
and opponent framework. These measurements controlled progress and candidate timing but are distinct from the
terminal result matrix.

The terminal matrix evaluated the selected 14×160 checkpoint over 100 games per row from 50 paired openings, with
each opening played from both colours. Stockfish 13 used one thread, 1,024 MiB hash, and fixed node budgets from the
published anchor curve. The reported rung for each model budget is the one whose observed score is closest to 0.5;
both rungs and confidence intervals appear in Appendix B. Searched play used the frozen INT8 TensorRT artifact.
Policy-only play used the matching float TorchScript export because it bypasses the native search service.

The measured curve spans policy only and 100, 1,000, 10,000, and 100,000 searches per move. Parallelism is one at
100 and 1,000 searches, four at 10,000, and sixteen at 100,000. Chapter 8 interprets the resulting operating curve
and its protocol-specific benchmark Elo.

## Interpretation

The terminal matches measure the assembled recipe. Component studies in Chapters 4--6 explain the retained choices;
Chapter 8 reports the resulting playing strength and training volume.
