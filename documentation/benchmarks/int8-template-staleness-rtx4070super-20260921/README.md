# INT8 refit-template staleness: V90 collapse and V91 recovery

Node `38.49.42.120:53893`, 8x RTX 4070 SUPER, driver/runtime as provisioned for V89.
All numbers below come from the same node and the same run directory, so they are directly comparable.

- Run directory: `py/training_data/production/vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau`
- V91 source revision `3e26ca517fcb9d63488601fdd3c42ec5c79bd41d`,
  `experiment_configuration_sha256` `be3afa9a0d3fe86878d44be782a4bc824fd13522548f261ed9fc7fdc3e79f176`
- Model under test: `chess-cnn-scaled-post-14x160-fromto-int8`, QAT phase `pre_fold`

## What happened

The 14x160 candidate was promoted at generation 481. Its INT8 serving engine was refit from
`/workspace/tensorrt/automatic-refit/b320-8467b8ab....engine`, a template built on 17 September,
roughly 124,000 optimizer steps earlier. A refit template fixes its quantisation state at build time
and `refit_engine` replaces weights only, so the served engine was quantised for a long-obsolete
checkpoint. `onnx_graph_signature` normalises Q/DQ constants to shape-only, so the cached template
matched forever and nothing rebuilt it.

Float training loss stayed the better of the two models throughout. A model cannot be hundreds of
Elo weaker in float while training better, and that asymmetry is what identifies a broken export
rather than a bad model.

## Fidelity, same run directory

`policy top1 / policy mean KL / policy max KL`, from the `TensorRT fidelity warning` log line.

| checkpoint | model | template | top1 | mean KL | max KL |
|---|---|---|---|---|---|
| 480 | 12x128 | 18 Sep | 0.7906 | 0.01848 | 0.09046 |
| 481 | 14x160 | 17 Sep | 0.1344 | 0.52173 | 2.05833 |
| 482 | 14x160 | 17 Sep | 0.1781 | 0.41544 | 1.30421 |
| 483 | 14x160 | 17 Sep | 0.1625 | 0.38884 | 1.46424 |
| 484 | 14x160 | 17 Sep | 0.1938 | 0.41024 | 1.01212 |
| 485 | 14x160 | 17 Sep | 0.1781 | 0.44096 | 1.45210 |
| **482** | 14x160 | **rebuilt from 482** | **0.9313** | **0.00155** | **0.00626** |

The last row is the controlled comparison: the same `model_482.int8.onnx`, carrying its own freshly
recalibrated scales, through a stale template and a rebuilt one. 5x on top1, 268x on mean KL.

V91 after the rebuild, template built from 482:

| checkpoint | top1 | mean KL | max KL |
|---|---|---|---|
| 486 | 0.8875 | 0.00223 | 0.00831 |
| 487 | 0.9063 | 0.00196 | 0.00940 |
| 489 | 0.9344 | 0.00245 | 0.00674 |
| 490 | 0.9000 | 0.00220 | 0.00584 |
| 492 | 0.8938 | 0.00253 | 0.01236 |
| 493 | 0.8781 | 0.00342 | 0.01094 |

## Playing strength, adaptive Stockfish ladder, 64 searches

Score is the candidate's score over the paired opening suite.

| generation | state | n=2000 | n=3000 | n=5000 | n=10000 |
|---|---|---|---|---|---|
| 457 | 12x128 | | 0.675 | 0.445 | 0.200 |
| 463 | 12x128 | | 0.615 | 0.465 | 0.205 |
| 469 | 12x128 | | 0.715 | 0.430 | 0.200 |
| 475 | 12x128 | | 0.650 | 0.470 | 0.195 |
| 481 | 14x160, stale template | | **0.160** | **0.025** | 0.025 |
| 486 | 14x160, rebuilt template | 0.820 | **0.720** | 0.460 | |
| 493 | 14x160, rebuilt template | | | **0.530** | |

Single-rung Elo at 3000 nodes: 12x128 at 0.650 is +108 over that rung, the stale-template 14x160 at
0.160 is -288, and the rebuilt 14x160 at 0.720 is +164. The defect cost about 450 Elo; the promotion
itself is worth about +56 Elo over the model it replaced at that rung, and about +42 at 5000 nodes.

## Recalibration is real but does not reach the engine

Comparing every QuantizeLinear/DequantizeLinear scale initializer between `model_486.int8.onnx` and
`model_487.int8.onnx`: **112 of 112 scale tensors changed**, the largest by 33% relative in a single
generation. `recalibration_interval_generations` is 1 and it is working. Those scales reach the ONNX
and do not govern the engine, which is what the controlled row above demonstrates.

Whether TensorRT ignores refitted scale constants outright, or applies them while retaining kernel
and fusion choices made for the old magnitudes, is not separated by these measurements.

## Limits

`MINIMUM_POLICY_TOP1_AGREEMENT` was 0.90 with mean and max policy KL at 1e-3 and 0.01. A template
rebuilt from the very checkpoint it serves measures 0.00155 mean KL, so the mean limit sat below the
healthy case and every export failed it. `allow_fidelity_deviation` was therefore set on every run
and the warning carried no information: the 0.42/2.06 collapse logged the same line as a healthy
export. Limits are now 1e-2 and 0.05, six times above the healthy case and an order of magnitude
below the broken one.

## Open

Top1 of 0.88 to 0.93 is itself low for INT8 against a float reference; ~0.99 would be expected unless
the probe positions have flat policies where argmax is noise-dominated. Whether that is genuine
quantisation error or an artifact of the metric is under investigation, as is the rebuild cadence:
top1 fell from 0.934 at generation 489 to 0.878 at 493 on a template built at 482.
