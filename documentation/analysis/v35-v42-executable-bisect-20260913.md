# V35 to V42 executable bisect

Date: 2026-09-13

This is the time-boxed executable follow-up to the full source audit. The online endpoint control uses V35-era
runtime code with V42's exact generation-zero checkpoint. This document isolates the remaining source boundaries
and defines the next online arm without changing the running endpoint control.

## Immutable control inputs

- V35 successful source: `8b02c00af25aafc4fe1edb3e16ec2fe5dfb3afa5`.
- V35 runtime plus repaired evaluation: `1d57b0ab853adef2764e864e87898de1c74f0a37`.
- V42 source: `22fa780b8f29bdbbb357b84125eb6e8faa35ff1a`.
- V42 generation-zero model SHA-256: `6f8a24e5c235c22097d4186ed3dd645089fa8fbc851241406dcf439faa987476`.
- Native TensorRT module SHA-256: `ccc023266e08c372327ead5cf7af0b49d70c00b2412a39ade7fbebaf84e896b0`.
- Pre-fold 14x160 template SHA-256: `40c9d086b6300f563c74f3a310d9d5181d26b360bfe81406325ac2affd4278cf`.
- Deployment 14x160 template SHA-256: `f6c083535365d3d6bb53487d0d4c33dd8ae5d69e33766b3f74d742275fcda52e`.

The first online control was operationally invalid because its projected native module lacked TensorRT. The r2
control replaced it with the pinned module above and passed a real TensorRT refit and native inference smoke before
launch. It is the only endpoint result used here.

## Boundary results

### Typed TensorRT template selection (`509eeda3`)

The parent and child configuration tests pass (`6/6` and `8/8`). The old ordered scan and new typed selection resolve
the same actual engine templates. Engine publication metadata from both V42 and the old-code r2 endpoint records the
same pre-fold and deployment template hashes listed above. The generated engine filenames and serialized engine
hashes differ because their source ONNX weights and cache-key implementation differ; that is expected. There is no
evidence that the child selects a different graph.

Classification: **cleared as a training regression unless a future same-ONNX output comparison fails**. It cannot
change model optimization, replay sampling, or native search after the selected engine is the same.

### Phase-specific post-fold warmup (`a10393ca`)

V35's absolute 1,000-step warmup is complete at the exact 1,000-step fold boundary. V42 explicitly configures zero
deployment warmup. Both therefore apply the configured deployment learning rate immediately after folding. The
focused current tests cover pre-fold, deployment, and disabled-QAT phase selection.

Classification: **cleared for the V35/V42 resolved settings**. It is a real feature but the two resolved schedules
are equivalent at every post-fold optimizer step.

### Live QAT sidecar identity (`304cd00b`)

The commit assigns the sidecar identity returned by checkpoint persistence back to the live runtime. It changes the
path and digest of the saved ModelOpt metadata reference. The function does not mutate model tensors, optimizer
state, quantizers, ONNX outputs, replay, or learning rate. The change prevents later checkpoint retention from
deleting a sidecar still referenced by the live runtime.

Classification: **cleared as a healthy uninterrupted-run learning regression**. It remains a checkpoint durability
fix. A failure would present as a missing-sidecar resume or publication error, not silently weaker updates.

### Warmup floor (`382442f7`)

This is the only boundary that changes a fixed V42 model's pre-fold optimizer updates. With peak learning rate 0.1
and 1,000 warmup steps:

| Completed step | V35 floor 0 | V42 floor 0.001 |
| ---: | ---: | ---: |
| 1 | 0.000100 | 0.001099 |
| 10 | 0.001000 | 0.001990 |
| 100 | 0.010000 | 0.010900 |
| 500 | 0.050000 | 0.050500 |
| 1,000 | 0.100000 | 0.100000 |

The largest relative difference is confined to the earliest updates. It is not a coding error: focused warmup/QAT
tests pass and the formula reaches both endpoints exactly. It can still change an initialization-sensitive online
trajectory, so it is the only source boundary worth an Elo arm.

Classification: **open empirical suspect; low prior for explaining a large persistent deficit**.

## Prepared online arm

Branch `codex/v42-warmup-bisect`, commit `5e72959f`, adds two matched current-head controls. Both import the immutable
V42 generation-zero checkpoint, use empty independent replay and output directories, retain the V42 14x160 topology,
and evaluate policy-only and 64-search strength every 1,200 seconds.

| Configuration | Only semantic arm value | Resolved SHA-256 |
| --- | ---: | --- |
| `vast-chess-8gpu-v42-g0-warmup-floor-zero-ab.yaml` | floor `0.0` | `fd2393d0bc14fc4e22caf0535be0873cdc825eecc8a5e648fba04f38652fe01b` |
| `vast-chess-8gpu-v42-g0-warmup-floor-001-ab.yaml` | floor `0.001` | `b8522f367872cc8d5b64fed597600357d2c0e7211c7fd2602b044d4eea99e33a` |

The `0.001` arm is a reproducibility specification for V42. V42's existing run already supplies its endpoint, so
the next GPU hour should be spent on the floor-zero arm rather than repeating both arms.

## Decision within the one-hour deadline

1. Finish the running old-code r2 endpoint through its 60-minute evaluation.
2. If old code remains on V42's weak trajectory, reject a post-V35 code regression. Keep the progressive and typed
   TensorRT work. Treat the original V35 acceleration as an initialization/online-trajectory result and use the
   deterministic sharp-policy initialization fix for future comparisons.
3. If old code reproduces V35's acceleration, stop broad source speculation and run the prepared current-head
   floor-zero arm. Recovery there attributes the difference to `382442f7`; failure promotes the typed-template
   boundary despite its matching artifacts and requires a same-ONNX native-output test.

The project has already established that uniform legal-action initialization learns slowly. This bisect does not
propose or test uniform initialization. Deterministic initialization should preserve the calibrated sharp prior.

## Validation

- Current-head focused remote suite: `33 passed` across trainer warmup, QAT lifecycle configuration, and self-play
  configuration tests.
- Typed-template parent: `6 passed`.
- Typed-template child: `8 passed`.
- Both prepared configurations resolve locally; their hashes are recorded above.
- Local broad configuration collection is blocked because the workstation does not install `nvidia-modelopt`; the
  locked remote environment supplied that dependency and passed the focused suite.

No GPU benchmark was run while the online endpoint occupied all eight GPUs.
