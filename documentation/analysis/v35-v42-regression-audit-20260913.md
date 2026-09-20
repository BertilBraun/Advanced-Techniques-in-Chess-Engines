# V35 to V42 online-learning regression audit

Date: 2026-09-13

Audited range: `8b02c00af25aafc4fe1edb3e16ec2fe5dfb3afa5..22fa780b8f29bdbbb357b84125eb6e8faa35ff1a`

Successful V35 training source: `8b02c00af25aafc4fe1edb3e16ec2fe5dfb3afa5`

V35 evaluation-fix source: `1d57b0ab853adef2764e864e87898de1c74f0a37`

V42 source: `22fa780b8f29bdbbb357b84125eb6e8faa35ff1a`

## Conclusion

The source diff contains no demonstrated post-V35 defect that can explain V42's weak first 80 minutes. The network
implementation and all production C++ are byte-identical across the range. The resolved V35 and V42 experiments use
the same fixed 14x160 network, SGD/NAG optimizer, batch sizes, fold point, post-fold learning rate, replay policy,
self-play topology, search schedule, openings, objectives, and auxiliary targets.

Only two intentional differences execute in fixed V42 and can affect the experiment before its first four
evaluations:

1. V42 raises the first optimizer-step learning rate from `0.0001` to `0.001099` by adding a `0.001` warmup floor.
   It reaches the same `0.1` at step 1000. This is the strongest training-semantic suspect because it changes every
   pre-fold update. It points in the opposite direction from a "too little learning" explanation, but a ten-times
   larger first step can put the optimizer on a different early trajectory.
2. V42 evaluates every 1200 seconds instead of every 1800 seconds. Its completed policy plus searched evaluations
   consumed about 70-89 GPU-seconds per boundary on one GPU. This is about 0.8% of total eight-GPU capacity versus
   about 0.6% in V35. It can slightly reduce wall-clock throughput, but cannot explain poor strength per generation.

The TensorRT template-selection refactor also executes, but the actual V35 and V42 medium-model engine templates are
byte-identical for all three relevant identities. It therefore changes which file is selected explicitly, not the
engine content. The remaining production changes are evaluation-only, progressive-session-only, failure tolerance,
serialization, or checkpoint-sidecar durability. None changes fixed V42's training tensors or native search
algorithm in the healthy path.

There is one material pre-existing confound: generation-zero model creation ignored `training.random_seed`. V35 and
V42 therefore started from different random weights despite identical configured seeds. This bug predates V35 and
is not a V35-to-V42 regression. It prevents an uncontrolled comparison from attributing the early-strength gap to
source changes. The controlled old-code run using V42's exact generation-zero checkpoint is the right discriminator.

## Evidence and comparison basis

The archived V35 resolved configuration has SHA-256
`3e4816f042fce8bb079af28ed5e365957591529b8f4a7827163c7ddbbca69482`. The live V42 resolved configuration has
SHA-256 `755025f941aa4f3bf47f154f9b6005957cc6f7b7c39d03cd5ee987e2d79cb355`. They were read from:

- `/workspace/alphazero-engine-int8-native/.codex-diagnostics/vast-chess-8gpu-integrated-v35-int8-from-scratch-20260913T082213Z/run/resolved-experiment.json`
- `/workspace/alphazero-engine-int8-validation/py/training_data/production/vast-chess-8gpu-fixed-medium-v42-int8/resolved-experiment.json`

The audit used resolved configurations rather than authored inheritance because the V35 source configuration was
amended during its startup and evaluation repair. The following behavior is identical in the two resolved files:

| Surface | V35 and V42 resolved behavior |
| --- | --- |
| Model | Fixed `chess-cnn-scaled-post-14x160-fromto-int8`; 14 layers, 160 channels, scaled post-activation branch scale `0.2672612419124244`, cap `6.0`, global pooling every second block, from-to policy key 128, value channels 2, value FC 48 |
| Optimizer | SGD, momentum `0.9`, weight decay `0.0001`, Nesterov enabled |
| Training topology | Eight DDP ranks on GPUs 0-7, global batch 2048, local batch 256, BF16, compilation disabled, prefetch depth 4, gradient clipping 1.0 |
| QAT | Fold at optimizer step 1000; 256 calibration positions; recalibrate every generation; deployment LR linear `0.02 -> 0.01`, generations 2-1000 |
| Replay | Ratio 6.25; 500 optimizer steps per quantum; staged capacity `450k, 900k, 1.5M, 2.1M, 3M, 4.5M, 6M, 9M, 12M, 15M`; policy-surprise sampling with 0.3 uniform probability and cap 2.0 |
| Self-play topology | Four processes per GPU, 512 concurrent games per process, two of four processes per GPU paused during training |
| Inference | TensorRT, batch 320, one inference worker, two outstanding batches, TorchScript generation-zero bootstrap |
| Search and starts | Visits `300, 400@10, 500@50, 600@90, 800@1000`; the same parallel-search schedule; 50% random 0-8-ply starts and 50% regret restarts |
| Objective | Policy:value 1:1, same root-value blend and auxiliary next-policy and remaining-length heads |
| Evaluation games | Both use 50 paired openings per definition, Stockfish 13, and the same node ladder for policy-only and 64-search evaluations |

The resolved differences are:

| Surface | V35 | V42 | Expected effect |
| --- | --- | --- | --- |
| Base LR horizon | `0.1 -> 0.01`, generations 0-1000 | `0.1 -> 0.01`, generations 0-1200 | Before the step-1000 fold this changes generation 1 from about 0.09991 to 0.099925; negligible. After fold the explicit deployment schedule overrides it. |
| Initial warmup | 1000 steps, implicit floor 0 | 1000 steps, floor 0.001 | Material pre-fold trajectory difference. |
| Deployment warmup | No separate field; absolute global warmup was already complete after fold | Explicit zero steps, floor field unused | Equivalent. |
| TensorRT configuration | Ordered untyped V35 paths | Typed V36 paths for model and QAT phase | Selection is more explicit; medium engine bytes are identical. |
| Evaluation cadence | 1800 s, first generation 2 | 1200 s, first generation 0 | More points and slight GPU contention. Generation zero is still evaluated only at the first due boundary's active checkpoint. |

No diff exists in `py/src/training/network.py`, `cpp/src`, `cpp/include`, or
`py/tools/publish_tensorrt_engine.py`. The native test-only change in `9d21a258` does not modify the extension.

Current-source anchors for the audited runtime surfaces are:

| Surface | File and line at `22fa780b` |
| --- | --- |
| Chess evaluation specialization | `py/src/games/chess/training.py:105` |
| Checkpoint-aware model/template resolution | `py/src/games/implementation.py:134`, `py/src/games/implementation.py:153` |
| Typed TensorRT backend | `py/src/self_play/configuration.py:68` |
| Native TensorRT refit dispatch | `py/src/self_play/native_configuration.py:68` |
| Self-play checkpoint resolution | `py/src/self_play/worker.py:147` |
| Warmup floor configuration | `py/src/training/configuration.py:157` |
| Progressive QAT candidate cleanup | `py/src/training/progressive.py:502`, `py/src/training/progressive.py:522` |
| QAT phase warmup | `py/src/training/quantization/configuration.py:59` |
| Evaluation ONNX specialization | `py/src/training/quantization/checkpoint.py:151`, `py/src/training/quantization/runtime.py:206` |
| Self-play response timeout | `py/src/training/self_play_group.py:22` |
| Progressive fold restart | `py/src/training/session.py:281` |
| Warmup formula and live sidecar update | `py/src/training/trainer/rank.py:327`, `py/src/training/trainer/rank.py:517` |
| Unseeded checkpoint-zero construction | `py/src/experiment/run.py:232` |

The medium TensorRT templates on the node were checked directly:

| Template | V35 SHA-256 | V36/V42 SHA-256 |
| --- | --- | --- |
| Pre-fold, batch 320 | `40c9d086b6300f563c74f3a310d9d5181d26b360bfe81406325ac2affd4278cf` | same |
| Deployment, batch 320 | `f6c083535365d3d6bb53487d0d4c33dd8ae5d69e33766b3f74d742275fcda52e` | same |
| Deployment evaluation, batch 64 | `e89029562c798d5e9e6887bb83eccd2890343c66b5d8e82da04c1135822507c1` | same |

V42's checkpoint manifests show the intended phase transition: generation 1 records `pre_fold` at 500 optimizer
steps, and generation 2 records `deployment` at 1000 steps. Its generation-zero model has SHA-256
`6f8a24e5c235c22097d4186ed3dd645089fa8fbc851241406dcf439faa987476`; the controlled seed copy has the same hash.

## Observed early evaluation

The scores below are 100-game adaptive-rung scores and have binomial-scale noise. They establish that V42 was weaker
in this sample, not why.

| Elapsed | V35 generation | V35 policy / searched | V42 generation | V42 policy / searched |
| ---: | ---: | ---: | ---: | ---: |
| 20 min | - | - | 2 | 0.055 / 0.090 |
| 40 min | - | - | 10 | 0.090 / 0.110 |
| 60 min | 21 | 0.110 / 0.135 | 18 | 0.070 / 0.110 |
| 80 min | - | - | 25 | 0.055 / 0.085 |
| 90 min | 37 | 0.205 / 0.470 | - | - |

The initial V35 30-minute evaluation failed before `d0d807b`; the first valid V35 point is at 60 minutes. V35's
largest early jump occurred between 60 and 90 minutes. Consequently, a controlled arm must run through at least the
90-minute boundary before being called from Elo alone. V42's 60-minute searched deficit is 2.5 games and its policy
deficit is 4 games. Those are suggestive, but not enough to diagnose a code regression independently of its 80-minute
reversal and different initialization.

## Production hunk audit

### `d0d807b` - evaluation artifact specialization

- `py/src/games/chess/training.py` stops rebuilding a QAT model for the evaluation batch size and calls a graph-only
  specializer.
- `py/src/training/quantization/checkpoint.py` validates the retained checkpoint ONNX artifact and specializes it
  instead of loading the raw training checkpoint, optimizer, and QAT sidecar.
- `py/src/training/quantization/runtime.py` adds `specialize_qat_onnx_batch`, which rewrites fixed batch dimensions,
  policy-head reshape constants, and scatter constants.

Fixed V42 executes this only for evaluation inference at batch 64 and generation greater than zero. Self-play keeps
the exported batch-320 ONNX unchanged. More importantly, all valid V35 evaluation points were produced after the
run resumed on `1d57b0ab`, which contains this commit. It therefore cannot explain a valid V35-versus-V42 training or
evaluation difference. If evaluation itself remains suspect, compare the two batch-64 ONNX paths from one immutable
checkpoint and require matching logits/WDL plus a short same-opening match.

### `ff06cbb` - progressive QAT fold support

- `py/src/training/configuration.py` permits multiple scaled post-activation models under QAT instead of requiring a
  fixed model.
- `py/src/training/progressive.py` retains a candidate checkpoint's QAT sidecar during cleanup.
- `py/src/training/session.py` restarts a progressive candidate's trainer exactly at its fold boundary.

V42 has `progressive_model_sizing.kind: fixed`. `create_training_session` therefore instantiates
`FixedTrainingSession`, so the progressive candidate retention and restart paths do not execute. The broader schema
validation accepts V42 but does not change it. Exclude this commit from the first fixed-model bisect.

### `509eeda` - typed template selection

- `py/src/self_play/configuration.py` replaces an untyped ordered path tuple with typed float/QAT templates keyed by
  model id and QAT phase.
- `py/src/games/implementation.py` reads the checkpoint manifest, requires its architecture to identify exactly one
  configured model, and resolves the appropriate template.
- `py/src/self_play/native_configuration.py` gives the publisher one selected template rather than asking it to scan
  a list for a compatible template.
- `py/src/games/chess/training.py`, `py/src/games/go/training.py`, and `py/src/self_play/worker.py` pass the complete
  checkpoint through this resolution path.

This executes for fixed V42. It is a plausible boundary for a path/phase-selection error, but current evidence
refutes the obvious forms: generation 1 and 2 have the expected phases; the configured medium templates are
byte-identical to V35's; the network architecture is unchanged; and the publisher implementation is unchanged. A
minimal test is to refit one V42 pre-fold and one deployment checkpoint through old list scanning and new typed
selection, then compare the selected template SHA, generated engine refit manifest, and native outputs exactly.

### `a10393ca` - phase-specific warmup

- `py/src/training/quantization/configuration.py` adds a deployment LR choice of an explicit schedule or `inherit`,
  and a separate deployment warmup length.
- `py/src/training/trainer/rank.py` chooses LR and warmup progress by QAT phase.
- `py/src/training/quantization/__init__.py` exports the new helper.

Fixed V42 executes this. Its deployment warmup is zero and its explicit deployment schedule is the same schedule V35
used. Under V35's absolute-step warmup, the 1000-step warmup was already complete exactly when the model folded.
Therefore the effective learning rate is equivalent. A tensor-level two-quantum test across the fold should confirm
that old and new implementations produce the same generation-2 weights when the warmup floor is held at zero.

### `9ad7e961` - progressive candidate cleanup

The production hunk in `py/src/training/progressive.py` deletes stale candidate `qat_state_*` artifacts. It only
applies to progressive candidate directories. Fixed V42 does not execute it. The benchmark-tool change is not in the
training runtime.

### `ea4d9d74` - worker startup tolerance

`py/src/training/self_play_group.py` raises the response timeout from 120 to 300 seconds. The corresponding
configuration edit is a type-annotation/validation surface. In a healthy run it changes no search, batching, model,
or replay output. It only avoids retiring a worker that takes 120-300 seconds to answer. A regression here would be
visible as worker retirement/restart logs, not silently weaker play.

### `ec16907f` - canonical backend serialization

`py/src/self_play/configuration.py` replaces the wrap serializer with an explicit JSON serializer so typed TensorRT
templates serialize cleanly while default TorchScript execution knobs remain omitted. This changes canonical JSON
construction, not the already parsed in-memory fields. Both resolved configurations were inspected and V42 contains
the intended backend, batch, precision, memory-format, and cuDNN values. No evidence indicates a missing runtime
setting. A serialization regression is tested by round-tripping the resolved V42 configuration and comparing the
typed object and native inference configuration.

### `304cd00b` - live QAT sidecar identity

`py/src/training/trainer/rank.py` updates `runtime.qat_state` to the just-saved checkpoint sidecar after publication.
The saved bytes are a copy of the same ModelOpt state; the model, optimizer, quantizers, and recalibration loop are
unchanged. This prevents a live runtime from continuing to reference an older sidecar that checkpoint retention may
delete. It is primarily restart/durability correctness and should not affect uninterrupted learning. The minimal
test is one identical post-fold replay quantum on each parent/child revision, comparing raw model and optimizer
tensors, ONNX outputs, phase, and completed steps. A difference in tensors would promote this suspect; a path/hash-only
difference would clear it.

### `382442f7` - warmup floor

- `py/src/training/configuration.py` adds `training.trainer.warmup_start_learning_rate`.
- `py/src/training/quantization/configuration.py` carries an initial and deployment warmup floor with phase-specific
  progress.
- `py/src/training/trainer/rank.py` changes warmup from `base * (step + 1) / steps` to
  `floor + (base - floor) * (step + 1) / steps`.

This is the only unequivocal change to fixed V42's pre-fold parameter updates. With base 0.1 and 1000 steps, V35's
first step is 0.0001, while V42's is 0.001099. Both reach 0.1 at step 1000. Test it first by running current code from
the exact V42 generation-zero checkpoint with only the floor changed from 0.001 to 0.0. Deployment floor is irrelevant
because deployment warmup is zero.

## Complete intervening commit ledger

Every commit in the audited range appears below. “Runtime” means code reached by fixed V42; “offline” means build,
benchmark, or experiment tooling; “configuration” means the commit defines an arm but does not change shared runtime.

| Commit | Files or hunk | Fixed V42 classification |
| --- | --- | --- |
| `d0d807b` | Chess evaluation creation; QAT checkpoint specialization; ONNX batch specializer; lifecycle tests | Runtime, evaluation only; also used by valid V35 evaluations |
| `1d57b0ab` | V35 resume-at-generation-16 configuration | Configuration only |
| `ff06cbb` | Training validation; progressive QAT sidecar retention; progressive trainer restart at fold | Progressive-only runtime; inactive |
| `509eeda` | Typed templates in self-play config; checkpoint-aware selection in game implementations, native configuration, and worker; production/validation config migrations | Runtime selection path; artifacts verified identical |
| `a10393ca` | Phase-specific LR/warmup config and trainer logic; tests/config | Runtime; effective post-fold values equivalent |
| `9ad7e961` | Progressive candidate QAT cleanup; native-loop benchmark; tests | Progressive-only runtime plus offline tooling; inactive |
| `f6eaff1c` | V36 production configuration and design note | Configuration/documentation only |
| `a30a9a5c` | Deterministic QAT template ONNX exporter | Offline template tooling only |
| `6a463eba` | Cached/bounded TensorRT template build tool | Offline template tooling only |
| `fb03f245` | Progressive lifecycle smoke configuration | Validation configuration only |
| `ea4d9d74` | Self-play response timeout 120 -> 300 seconds; configuration tests | Runtime failure tolerance; no healthy-path output change |
| `34a5e549` | Fresh smoke configuration | Validation configuration only |
| `ec16907f` | Typed TensorRT backend serializer | Runtime serialization boundary; resolved object verified |
| `b55406ef` | Keep pre-fold template QAT nodes refittable in exporter | Offline template tooling only |
| `5020749b` | Seed template batch norm after ModelOpt conversion | Offline template tooling only |
| `2bc75cc5` | Mark only ONNX-owned TensorRT weights refittable | Offline template tooling only |
| `07e54ebf` | Exclusive progressive smoke configuration | Validation configuration only |
| `304cd00b` | Replace live QAT sidecar identity after checkpoint save | Runtime durability; no expected tensor change |
| `438f6057` | Ordered lifecycle smoke configuration | Validation configuration only |
| `c8a554d2` | Pre-fold topology smoke configuration | Validation configuration only |
| `2f9f11b8` | Ordered promotion smoke configuration | Validation configuration only |
| `024a3cca` | Small-model evaluation-trigger smoke configuration | Validation configuration only |
| `28cb6411` | Resumable progressive smoke configuration | Validation configuration only |
| `4c85872d` | Self-play worker test-double update | Test only |
| `76866240` | Complete backend test double | Test only |
| `0c124a30` | Progressive INT8 benchmark results and raw evidence | Documentation/evidence only |
| `9d21a258` | Native refresh test update | Native test only; no production C++ |
| `10d0fbb5` | V36 LR horizon 1200 -> 1000 | Superseded configuration only |
| `ee8afb89` | V36 LR horizon 1000 -> 1200 | Configuration only |
| `e766ded6` | V37 high-rate configuration and config test | Configuration/test only |
| `a860058b` | V37 stop-limit nesting fix | Configuration only |
| `13dd8b4e` | V38 direct-rate configuration and config test | Configuration/test only |
| `382442f7` | Warmup floor schema, phase propagation, trainer formula, V39 config, tests | Runtime training difference; primary code suspect |
| `b6b44698` | Expected configuration hash update | Test only |
| `efb9ab5a` | Matched DDP replay-screen tool | Offline experiment tooling only |
| `2bc50ae7` | Supervisor wrapper and replay-screen tool adjustment | Deployment/offline tooling only |
| `cb0e94db` | Replay-screen protocol | Documentation only |
| `d73d16cb` | Replay-screen raw results | Evidence only |
| `8038451c` | Replay-screen analysis | Documentation only |
| `12864a88` | Self-play benchmark telemetry and float export tool | Offline benchmark tooling only |
| `854d6d63` | V39 throughput analysis and raw evidence | Documentation/evidence only |
| `dda39067` | V40 production configuration | Configuration only |
| `8c7f6e9e` | V41 small-model experiment configuration | Configuration only |
| `22fa780b` | V42 fixed-medium experiment configuration | Configuration only; selects final resolved behavior |

## Pre-existing initialization confound

`py/src/experiment/run.py::_save_random_initial_checkpoint`, introduced in `0f61e2ffe`, constructed the model before
calling any seed function. Trainer ranks later call `torch.manual_seed(training.random_seed)`, but by then they load
the already-created generation-zero checkpoint. Every rank loads the same file, and DDP also synchronizes parameters,
so ranks within one run begin aligned; separate runs with the same configured seed do not begin from the same weights.

Observed bootstrap policy prior shapes illustrate the uncontrolled difference:

| Run | Model | Initial top-1 | Initial top-3 | Applied scale |
| --- | --- | ---: | ---: | ---: |
| V35 | 14x160 | 0.14225 | 0.31754 | 6.5073 |
| V42 | 14x160 | 0.07476 | 0.19344 | 13.4621 |
| V40 | 12x128 | 0.11345 | 0.27831 | 9.4618 |
| V41 | 12x128 | 0.07014 | 0.18173 | 15.8604 |

Bootstrap calibration scales logits to an aggregate top-3 mass target of 0.95. It does not make action rankings,
relative gaps, WDL outputs, or all per-position policy distributions identical. A larger scale is therefore evidence
of a flatter random policy before calibration, not by itself evidence that the network is defective. It can still
change early self-play materially because ranking and value randomness survive calibration.

The fix is commit `142294a003a45c6d76878615d87b8b665dd49960` on `codex/deterministic-checkpoint-zero`: seed immediately before
model construction and test same-seed tensor identity, identical calibration, and different-seed divergence. It is
correct future-run hygiene, but must be analyzed separately from the V35-to-V42 range.

## Native build reproducibility issue

The first controlled old-code arm exposed a separate operational defect: its projected `AlphaZeroCpp.so` SHA-256
`0cf19396c304622e5c29dedcd335a3e61b2b9f86c31044e80e349bfe7f669658` lacked TensorRT, while the known production
module SHA-256 `ccc023266e08c372327ead5cf7af0b49d70c00b2412a39ade7fbebaf84e896b0` supports TensorRT. Production C++
source is unchanged across the audit, so this was a build/cache/projection mismatch, not a source-level learning
regression. The r2 preparation replaced it with the known module and passed a real TensorRT refit plus native-output
smoke before launch.

Every bisect arm must use exactly the same recorded TensorRT-enabled native module, or rebuild with one pinned
TensorRT toolchain and record the resulting hash. Merely importing `AlphaZeroCpp` is insufficient; preflight must
refit an actual checkpoint into the selected template and execute native inference. Otherwise a failed arm can be
misread as a historical source regression.

## Ranked controlled A/B and bisect plan

All online arms must import the same immutable V42 generation-zero checkpoint (model SHA-256
`6f8a24e5c235c22097d4186ed3dd645089fa8fbc851241406dcf439faa987476`), use a unique empty save/replay/game/TensorBoard
directory, use the same native module and TensorRT templates, and schedule both evaluations every 1200 seconds. Do
not copy a replay store, credit ledger, self-play state, evaluation state, or elapsed-time state. Record source SHA,
resolved-config SHA, native SHA, checkpoint manifest SHA, and template SHAs for every arm.

Historical revisions originally reject a checkpoint resume at generation zero. Apply the same minimal harness-only
schema patch to every historical arm so `mode: checkpoint, generation: 0` is accepted. Keep that patch constant and
do not include it in the bisected behavior.

1. **Finish the endpoint control already prepared.** Use `1d57b0ab` rather than raw `8b02c00a`: it preserves V35
   training behavior while including the evaluation repair required to obtain valid 20-minute points. The r2 harness
   is commit `d7c461d0e0b3030db1bdcc338c04b9c52503e242` on `codex/v35-controlled-ab`, resolved configuration SHA-256
   `d28e3d17f12f7a16d9dd55e5c7ab6596fd0110240807336d8b72d6a6f66f8109`. Run through at least 90 minutes because
   the historical V35 jump occurred from 60 to 90 minutes.
2. **Isolate the warmup floor on current code.** Run `22fa780b` twice from the same checkpoint and empty state, with
   `warmup_start_learning_rate` 0.0 and 0.001. This is a one-field A/B and the highest-value next test. Keep the
   deployment floor unchanged because deployment warmup is zero.
3. **Isolate typed template selection.** Compare the parent of `509eeda` (`ff06cbb`) with `509eeda`, porting the same
   fixed V42 configuration into each schema. First perform a no-training refit/output comparison for generations 1
   and 2. Only run a 90-minute online arm if engine outputs or selected identities differ.
4. **Isolate phase warmup logic.** Compare `509eeda` with `a10393ca`, using floor zero and explicit values that make
   both effective schedules identical. Compare generation-1 and generation-2 model and optimizer tensors after a
   frozen replay quantum before spending on online self-play.
5. **Isolate live QAT sidecar retention.** Compare the parent of `304cd00b` with `304cd00b` for one post-fold frozen
   replay quantum. Escalate to online play only if tensors or exported outputs differ; path/hash identity alone is
   expected.
6. **Evaluate the batch-64 specialization only if score plumbing remains suspect.** Compare pre-`d0d807b` model
   reconstruction/export and post-`d0d807b` graph specialization on one immutable checkpoint. This cannot explain
   self-play learning and is common to valid V35/V42 evaluation, so it is last.

For efficient execution, use two layers. First, run deterministic frozen-replay quanta and compare model tensors,
optimizer tensors, QAT phase, ONNX outputs, selected template hashes, and generated-engine outputs. This cheaply
locates true semantic differences. Second, use 90-minute online arms only at boundaries where the first layer finds a
difference, plus the endpoint and warmup-floor controls. Online self-play remains stochastic even with one shared
checkpoint, so a single 100-game evaluation must not be treated as a deterministic bisect oracle.

The executable old endpoint configuration is
`py/configs/production/vast-chess-8gpu-v35-code-v42-g0-ab-r2.yaml` on commit `d7c461d0`. It extends a fixed 14x160
controlled configuration, imports `/workspace/controlled-initializations/v42-generation-0/checkpoint_0.json`, uses a
fresh output path, and explicitly sets `evaluation.cadence_seconds: 1200`. Its resolved test asserts both
`stockfish-searched` at 64 searches and `stockfish-policy-only` at one search, each starting at generation 2.

## Decision rule

- If the old-code/V42-seed endpoint learns like V35, a post-V35 source or resolved-config difference is real. Run the
  warmup-floor A/B first, then the typed-template boundary.
- If it learns like V42, the early V35/V42 gap is most likely initialization luck or online stochasticity, not the
  intervening production diff. Retain the deterministic-generation-zero fix and stop reverting unrelated progressive
  and TensorRT lifecycle work.
- If the old-code arm has normal strength per generation but worse generations per hour, inspect operational
  contention and native-module identity. If strength per generation is weak with comparable throughput, inspect the
  warmup trajectory and initialization-sensitive self-play distributions.

This audit does not alter or stop the running controlled r2 arm.
