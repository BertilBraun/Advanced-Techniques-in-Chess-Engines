# 6. Integrated chess recipe

## Living recipe and frozen result

The readable entry point for reproducing or extending the system is the fully expanded
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). It owns the current network ladder,
trainer, replay, self-play, deployment, and online-evaluation settings without an inheritance chain. It is a living
recipe: future project work may revise it.

The reported experiment is immutable. Its selected checkpoint, model and engine hashes, evaluation rows, archive
digests, and evidence status are recorded in the [final result](../results/final-chess-run.md) and
[evidence index](../evidence/final-chess-20260923/README.md). Those frozen records—not a future state of the YAML—are
the authority for published numbers.

## Recipe at a glance

| Component | Configured method |
| --- | --- |
| Compute | Eight RTX 4070 SUPER GPUs shared by self-play, training, and evaluation |
| Trainer | Eight-rank NCCL DDP; global batch 2,048; 500 optimizer steps per quantum; bfloat16 |
| Optimization | Nesterov SGD, learning rate 0.1→0.01, momentum 0.9, weight decay 0.0001, gradient-norm cap 1.0 |
| Deployment | Pre-fold INT8 QAT, real-position calibration, fixed-batch ONNX, TensorRT refitting |
| Network family | Scaled post-activation CNNs with global pooling, from-to policy, and WDL value head |
| Replay | Staged 0.6M→20M live rows, reuse ratio 4, policy-surprise sampling with 30% uniform mixing |
| Self-play search | Fixed staged visits, reduced-parent-value FPU, forced playouts, Dirichlet root noise |
| Starts | 50% shallow random openings and 50% recent restart states |
| Targets | Policy and WDL plus next-policy and remaining-game-length auxiliaries |
| Evaluation | Paired openings against fixed-node Stockfish 13 anchors; native search for searched conditions |

This table is explanatory. Exact schedules, paths, dimensions, and resource limits remain in the YAML.

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

Half of games begin after up to eight random legal plies. The other half use recent restart states filtered by value,
remaining length, age, and branchable visit mass; restart selection retains a 30% uniform component. Game caps grow
from 150 to 250 plies, and greedy move selection begins later as training matures. Calibrated resignation begins only
after sufficient evidence, constrains the false-nonloss upper bound to 2.5%, and continues 20% of triggered games for
ongoing safety measurement.

Replay capacity grows through ten stages from 600,000 to 20 million live rows. Each admitted position funds four
training presentations. Sampling reserves 30% uniform probability and otherwise prioritizes bounded policy surprise.
Eight materializers convert completed games into the fixed-layout memory-mapped store; credit is committed only
against durable admitted data.

## Progressive sizing

Candidate start and promotion are separate decisions. The primary 64-search ladder is smoothed and used to detect
stage-specific plateaus; the full thresholds and window semantics are documented in
[Progressive model sizing](../architecture/progressive-model-sizing.md). An eligible successor receives an average
of 1.5 optimizer quanta per active-model quantum on the captured replay snapshot and uses its own catch-up
learning-rate clock.

Promotion is decided by candidate-versus-active matches, not training loss. The configured gate requires a score of
at least 0.48 in two consecutive paired evaluations. This distinction matters because extra candidate presentations
made the earlier loss comparison systematically favorable to the candidate and admitted a much weaker model.

The completed capacity study also grew the trained 14×160 network into 19×176 while preserving its function, then
recovered deployment fidelity through QAT. The larger continuation reached parity but did not establish a stronger
plateau. The reported checkpoint is therefore the 14×160 model. This result says that capacity was not the immediate
bottleneck under this recipe; it does not establish a general limit on larger networks.

## Training-time and terminal evaluation

During training, evaluation ran every 20 minutes on 50 paired openings. The searched ladder used 64 searches per move
against a three-rung adaptive bracket of fixed-node Stockfish 13 opponents; a policy-only ladder used the same opening
and opponent framework. These measurements controlled progress and candidate timing but are distinct from the
terminal result matrix.

The terminal matrix evaluated the selected 14×160 checkpoint over 100 games per row from 50 paired openings, with
each opening played from both colours. Stockfish 13 used one thread, 1,024 MiB hash, and fixed node budgets from the
published anchor curve. The reported rung for each model budget is the one whose observed score is closest to 0.5;
both rungs and confidence intervals remain in the result record. Searched play used the frozen INT8 TensorRT artifact.
Policy-only play used the matching float TorchScript export because it bypasses the native search service.

The measured curve spans policy only and 100, 1,000, 10,000, and 100,000 searches per move. Parallelism is one at
100 and 1,000 searches, four at 10,000, and sixteen at 100,000. It is therefore an attainable operating curve, not a
single-variable search-scaling experiment. Absolute values are protocol-specific benchmark Elo and must not be
presented as FIDE ratings or unrestricted engine ratings.

## What the final result establishes

The terminal matches evaluate the assembled recipe. They do not allocate its strength gain among replay growth,
restart states, auxiliary targets, resignation, progressive sizing, or any other bundled choice. Some components
have isolated throughput, fidelity, proxy, or correctness evidence; fewer have controlled online strength ablations.
The investigation chapters state those boundaries individually.

Likewise, the `$43.20` reported cost is 60 hours of accepted-lineage time at the recorded hourly price through the
selected checkpoint. It excludes discarded work, later capacity experiments, distillation, terminal evaluation, and
idle rental time. It is a reproducible denominator for the reported checkpoint, not total project expenditure.
