# Small-model QAT audit on RTX 4070 SUPER

Date: 2026-09-13

## Conclusion

The 12x128 network, QAT fold, and production TensorRT refit all work. None reproduces V40's failure on a fixed
replay distribution. V40's poor online Elo is therefore most consistent with an online self-play bootstrap problem,
not a broken small architecture or corrupted folded inference.

The training implementation did not change between successful V35 source `1d57b0ab` and V40 source `dda39067`:
the Git blobs for both `network.py` and `training/quantization/runtime.py` are identical. TensorRT template creation
did change, chiefly to make BatchNorm and ONNX-owned weights refittable. A direct check of V40 generation 48 against
the exact refitted production engine found 97.44% legal-policy top-action agreement and policy KL 0.000846 over
3,200 held-out positions. Expected-value MAE was 0.0121. Those errors are far too small to explain an approximately
300-Elo early-training deficit.

## Matched replay screen

All arms used the same V34 replay, seed, sample-index sequence, global batch 2,048, 1,000-step warmup to 0.1,
fold at step 1,000, gradient clipping at 1.0, momentum 0.9, and weight decay 0.0001. Each arm used two RTX 4070 SUPER
GPUs. The three small arms completed 12,000 steps; the slower medium arm completed 9,575 within its 1,200-second
limit.

| Architecture and optimizer | Post-fold schedule | Step | Held-out total | Target top action | Mean gradient | Clipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 12x128 NAG | 0.02 immediately | 12,000 | 2.79899 | 46.95% | 1.822 | 100.0% |
| 12x128 NAG | 0.001 to 0.06 over 3,000 steps | 12,000 | **2.72090** | **49.32%** | 1.030 | 57.8% |
| 12x128 momentum | 0.001 to 0.06 over 3,000 steps | 12,000 | 2.72819 | 48.32% | 0.942 | 21.8% |
| 14x160 NAG | 0.02 immediately | 9,575 | 2.79702 | 46.51% | 1.903 | 100.0% |

At matched step 8,000, medium 0.02 reached total loss 2.81706 and 45.90% agreement. Small 0.02 reached 2.84784 and
45.31%; small 0.06 reached 2.79635 and 46.66%. The medium model has a small capacity advantage over the small model
under the same 0.02 schedule, but no qualitative advantage. The small model trains normally and the 0.06 schedule
is the strongest fixed-replay arm after its warmup. Ordinary momentum and NAG are close; NAG wins by 0.27% total
loss and one percentage point of target top-action agreement at step 12,000.

There is no fold collapse. Every arm improves substantially across the step-1,000 fold and continues smoothly.
For example, small 0.06 NAG moves from held-out total 3.29672 at the fold to 3.04844 at step 3,000, 2.85196 at
step 6,000, and 2.72090 at step 12,000.

## Initialization and causal interpretation

V35's logged training-forward initialization was policy-logit standard deviation 0.194 and entropy ratio 0.991.
V40's was 0.150 and 0.996. Both are healthy, nearly uniform priors. The difference follows from constructing a
different-sized trunk before the policy head, which advances the seeded random stream by a different amount, plus
the different trunk geometry. Generation-zero self-play uses a separately calibrated inference copy targeting the
same policy-prior shape, so this modest scale difference is not evidence of a defective small model.

The replay screen is supervised fitting to mature V34 labels. It cannot validate an LR against weak, moving labels
from a random self-play bootstrap. Its result rules out architecture and folding faults, but it does not rehabilitate
0.06 for online training. The only successful online SGD/QAT evidence remains V35's 14x160 model at 0.02. If the next
run must start 12x128, the clean discriminating run is 12x128 with the V35 0.02 post-fold schedule. If that also
fails online, model size or the generation-zero trajectory is causal; if it succeeds, V40's high online LR was
causal. Fixed-replay loss cannot resolve those alternatives.

## Reproducibility

- Replay-screen source: `15ee8243b7a5c14c32ccc0133c290c765772c750`
- Fidelity-tool source: `e6bc5ae1`
- Node: `38.49.42.120:53893`, 8x RTX 4070 SUPER
- Replay SHA-256: `d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83`
- Replay experiment SHA-256: `3c497f30e9e706b455f4fa3e18ba0b5caf892d2db2a7ada89093098af3b5ac98`
- V40 generation-48 ONNX SHA-256: `3bfbf139fdc6b34055b95953b86e0889cd5903146a2267b12a71210ecbc93d90`
- V40 generation-48 production engine SHA-256: `9aaa797a291532e9bb3b0a9f3f1ec954c80b432c12f471f02808ca5b27c906f1`

The four arms were started from the checked-in `supervisor.conf`. Each command resolves to:

```text
deployment/sgd_replay_screen_arm.sh ARM GPU_0 GPU_1 OUTPUT LAYERS HIDDEN_SIZE
```

Raw reports, observation CSV files, logs, and the deployed-engine fidelity report are under `raw/`.

Validation:

- `ruff format py/tools/run_sgd_replay_screen.py py/tools/audit_deployed_qat_checkpoint.py`
- `ruff check --fix py/tools/run_sgd_replay_screen.py py/tools/audit_deployed_qat_checkpoint.py`
- `python -m pytest --import-mode=importlib test/test_qat_lifecycle_runtime.py -q`: 9 passed
- Four real two-rank DDP arms completed without divergence.
