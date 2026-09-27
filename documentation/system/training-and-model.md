# Training and model

## Final model family

The final chess recipe trains a three-stage convolutional residual ladder:

| Stage | Residual tower | Policy head | Context |
| --- | --- | --- | --- |
| `chess-cnn-scaled-post-12x128-fromto-int8` | 12 blocks, 128 channels | from-to attention, key size 128 | global pooling every second block |
| `chess-cnn-scaled-post-14x160-fromto-int8` | 14 blocks, 160 channels | from-to attention, key size 128 | global pooling every second block |
| `chess-cnn-scaled-post-19x176-fromto-int8` | 19 blocks, 176 channels | from-to attention, key size 128 | global pooling every second block |

Every block uses capped scaled post-activation residuals; its branch scale is configured as the inverse square root
of the stage depth. The value head predicts win/draw/loss through a two-channel spatial reduction and a 48-unit
hidden layer. The training model also owns next-policy and remaining-game-length heads, but the published inference
artifact contains only the shared trunk, primary policy head, and WDL head. Exact construction is in
[`training/network.py`](../../py/src/training/network.py); the YAML remains the owner of dimensions and scales.

## Initialization and objective

Run preparation creates 20 seeded initialization candidates. It rejects candidates with an over-concentrated
policy or unhealthy WDL distribution, then calibrates the selected trainable policy scale toward 95% top-three mass
on 516 real probe positions. The coordinator repeats a policy-health guard against real replay data before the first
optimizer quantum.

The resolved loss for the final run combines:

- primary policy cross-entropy, weight `1.0`;
- WDL cross-entropy, weight `1.0`;
- next-ply policy cross-entropy, weight `0.15`, masked where no later search observation exists;
- remaining-game-length regression, weight `0.1`, normalized by 400 plies and censored for cut games.

The WDL target begins as the final game outcome discounted by `0.998` per ply. A scheduled blend with the stored
search-root value increases linearly from zero at generation 50 to `0.1` at generation 110. Search backups use their
separate `0.99` per-ply discount. Objective ownership and target variants are in
[`training/objective.py`](../../py/src/training/objective.py) and
[`training/targets.py`](../../py/src/training/targets.py).

## Optimizer and DDP quantum

All eight GPUs participate in one NCCL DDP group. Each rank consumes 256 positions per step for a 2,048-position
global batch. A quantum contains 500 optimizer steps. The optimizer is SGD with momentum `0.9`, Nesterov enabled,
and weight decay `0.0001`. The base learning rate warms from zero for the first 1,000 local optimizer steps, then
follows the run-wide linear schedule from `0.1` to `0.01` through generation 1,000. Training uses bfloat16
autocasting, clips the global gradient norm to `1.0`, and does not use `torch.compile`.

Trainer ranks are persistent processes and keep model, optimizer, CUDA context, and data-transfer machinery alive
between quanta. Rank zero alone writes checkpoints. The orchestration is split between
[`training/trainer/group.py`](../../py/src/training/trainer/group.py) and
[`training/trainer/rank.py`](../../py/src/training/trainer/rank.py).

## Quantization-aware training

The final run uses TensorRT INT8 QAT from the beginning. Convolution activations and weights are instrumented through
ModelOpt while linear layers, the input block, policy head, and value head remain outside the selected quantization
set. Calibration uses 516 positions from the immutable evaluation dataset and is refreshed every generation.

The configured fold boundary is one million optimizer steps, which is generation 2,000 at 500 steps per quantum.
The deployment-copy mode is important: if the boundary is reached, batch-normalization folding and recalibration
occur on a copied deployment model, while the trainable model and optimizer remain on the pre-fold parameterization.
This avoids changing the optimization problem merely to create an inference artifact. Generation zero may bootstrap
with TorchScript; from generation one the deployment path exports fixed-batch ONNX with explicit Q/DQ nodes and
publishes a refitted TensorRT engine. See
[`training/quantization`](../../py/src/training/quantization) and
[`self_play/native_configuration.py`](../../py/src/self_play/native_configuration.py).

## Progressive sizing

Only the active model trains initially. Candidate start is controlled by the searched Stockfish ladder and is
stage-specific: 15 Elo/hour for 14×160 and 4 Elo/hour for 19×176. The runtime uses a bias-corrected Elo EMA with
decay `0.90`. It retains seven EMA samples, measures gain across the six observation intervals between the oldest
and newest samples, and requires two consecutive complete-window gains strictly below the threshold before the
immediate successor starts. After a promotion, the next stage begins a fresh plateau state seeded from the latest
observed Elo rather than reusing the previous stage's latch.

Once eligible, the active model and its immediate successor train sequentially on the identical replay snapshot and
deterministic sample identity. The ordinary controller successor starts from its own random initialization; no
weights or optimizer moments transfer. The active model trains one optimizer quantum per global generation. The
successor's configured step multiplier is `1.5`, implemented as alternating one and two complete quanta from the
global generation index.
When two successor quanta run, both use the same global replay-source step and therefore repeat the same
deterministic batch sequence while the successor optimizer advances. Its catch-up learning rate follows its own
local-generation schedule from `0.1` to `0.01` through 600 local generations.

Promotion is match-based. At each evaluation boundary an eligible successor plays the published active checkpoint.
The candidate must score at least `0.48` in two consecutive completed paired matches; a failing score resets the
count, while a failed or cancelled evaluation adds no observation. The former loss-EMA gate was removed because the
candidate's extra quanta made its replay-fitting loss systematically incomparable with the active model's loss. Only
one checkpoint is published for self-play and evaluation after the entire global quantum. The implemented state
machine and recovery record are in
[`training/progressive.py`](../../py/src/training/progressive.py) and
[`training/session.py`](../../py/src/training/session.py).

The standalone final YAML includes the `progressive_candidate` evaluation referenced by the gate. It is the living
entry point for reproducing the current recipe; exact historical reproduction still requires the frozen resolved
configuration and source revision.

The completed capacity investigation also used a manual function-preserving transition from 14×160 to 19×176. New
units were wired random-in/zero-out, appended residual blocks were initialized as identities, branch scales were
compensated on batch-normalization affine parameters, and global-pooling channel roles were preserved. The probe
measured a `1.34e-05` maximum policy difference and exact top-one agreement. After one float replay-window epoch,
the grown network required ten QAT quanta to restore acceptable INT8 fidelity. This path is implemented by
[`grow_checkpoint.py`](../../py/tools/grow_checkpoint.py),
[`train_grown_checkpoint.py`](../../py/tools/train_grown_checkpoint.py), and
[`quantize_grown_checkpoint.py`](../../py/tools/quantize_grown_checkpoint.py); it is not yet the controller's
automatic initialization policy.

The small-to-medium transition worked reliably across the completed runs. The medium-to-large transition did not:
independently initialized candidates took too long to catch up, while the function-preserving continuation reached
parity but did not improve within its limited training window. The reported checkpoint therefore remains 14×160.
This result does not show that capacity was unimportant; it leaves training duration, post-growth optimization,
target quality, replay composition, and usable additional capacity confounded.

## Checkpoint publication and recovery

Each progressive model has a private checkpoint namespace. A complete checkpoint contains raw training weights,
optimizer state, QAT state, a trimmed deployment artifact, hashes, network definition, and policy-prior calibration.
The manifest is written last. `progressive-training.json` records the active model, each candidate's progress,
candidate-start state, match-gate evidence, and any pending quantum with its exact replay identity and completed
model prefix.

After every required model has trained, the selected active private checkpoint is published into the ordinary
generation namespace. The progressive state is completed and then the credit ledger commits the generation. On a
restart, a pending quantum resumes at its first incomplete model and refuses a changed replay identity. This is why
private candidate checkpoints are restart state, not directly consumable production models.
