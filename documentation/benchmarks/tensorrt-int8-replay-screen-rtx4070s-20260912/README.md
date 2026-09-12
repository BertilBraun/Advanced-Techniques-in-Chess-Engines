# TensorRT INT8 frozen-replay screen (RTX 4070 SUPER, 2026-09-12)

This experiment asked whether INT8 TensorRT inference could preserve the v34 learning target while materially increasing self-play throughput. It used the preserved 10,000,000-position v34 replay snapshot (`d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83`) and an untouched 2% holdout. All throughput numbers are isolated batch-320 model positions per second on one RTX 4070 SUPER; projected self-play gains use the measured production control of 48,463 positions/s/GPU.

## Decision

The strongest deployment graph is the production-sized 14x160 scaled post-activation model. A 1,000-step QAT model followed by BN folding and 5,000 replay steps in the exact deployment topology reaches 135,229 INT8 positions/s, versus 60,212 for TorchScript BF16 and 99,040 for TensorRT FP16. Its fake-quant replay loss is only 0.00069 above the quantizers-disabled model. End-to-end float-to-TensorRT fidelity is 96.12% policy top-1, policy KL 0.00141, WDL MAE 0.00445, and expected-value MAE 0.01096. Two independent 1,000-step smokes bracket the same throughput at 134,600–135,100 INT8 positions/s and show 96.54–96.93% policy agreement. The fixed-overhead model projects this to about **1.7x end-to-end self-play throughput**. TensorRT refit reduces the recurring engine update itself to 0.156 seconds, making this practical per generation; calibration and ONNX export remain roughly two seconds combined.

The full 100,000-step architecture screen and deployment-form recovery used the smaller 12x128 model. This establishes that the architecture can learn the replay target, but it is not a fully trained 14x160 candidate. After a 5,000-step folded continuation the 12x128 model reaches 94.76% policy top-1 agreement, policy KL 0.00348, WDL MAE 0.00819, and expected-value MAE 0.02010 before export. A further 10,000-step continuation is worse: 93.85% top-1, KL 0.00449, WDL MAE 0.00960, and expected-value MAE 0.02355. The 5,000-step checkpoint is therefore the retained recovery result.

The 12x128 replay-target proxy is considerably stronger than raw agreement suggests. On the same 51,200 untouched positions, the deployed 5,000-step INT8 engine has total target loss **2.54949**, compared with **2.55083** for the independently trained matched floating arm. The INT8 policy loss is 0.00315 higher, while its WDL loss is 0.00449 lower. TensorRT-versus-ONNX policy flips are concentrated at ambiguous decisions: agreement is 98.28% where the reference top-two logit margin is at least 0.05, 99.12% at 0.1, 99.89% at 0.25, and 100% at 0.5. Together with the production-size smokes, this supports integrating the backend for a bounded self-play experiment, but it does not substitute for chess games or a long 14x160 replay run.

No chess match was run because native search currently loads TorchScript only. A meaningful match requires a TensorRT inference provider in the C++ search runtime, fixed/dynamic batch execution and buffers, atomic engine refresh or refit, and artifact/config plumbing for the ONNX plan and refittable engine. Until that exists, the 1.7x system gain remains an Amdahl projection and chess strength remains unresolved.

## Architecture findings

- The original full-trunk PTQ graph reaches about 3.0x the TorchScript core rate, but its policy and WDL outputs are catastrophically wrong.
- The pre-activation quantization-friendly graph learns the replay target normally, including under QAT, but its TensorRT output is catastrophically wrong and its graph contains hundreds of layers and reformats.
- Scaled post-activation QAT learns normally and can be recovered after BN folding. At production size, explicit clipping and boundary conversions limit the INT8 advantage to 1.37x over TensorRT FP16; the smaller 12x128 graph reaches only roughly 1.09x because TensorRT FP16 uses that shape more efficiently.
- Removing the residual multiply and final clip restores residual fusions but leaves fused second convolutions producing FP16, so throughput barely changes.
- A single numerically shared residual-output scale produces faithful TensorRT output, but it is slower: 93,682 positions/s at 14x160, only 1.54x TorchScript and below the approximately 98,000 positions/s TensorRT FP16 control. Matching scales alone does not keep the residual path in INT8.

The remaining high-value architecture route would train directly in the compact, folded TensorRT topology with one Q/DQ boundary dominating every residual tensor before fan-out, while ensuring that the residual convolution, skip and post-add activation share an INT8 format. This screen did not demonstrate such a graph at useful speed.

## Optimizer calibration

A five-minute, 5,000-step frozen-replay probe used the same sampled indices, batch 1,024, constant learning rate after 200 warm-up steps, momentum 0.9, Nesterov and weight decay `1e-4`:

| Optimizer | LR | Held-out policy | Held-out WDL | Held-out total |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 0.002 | 1.98499 | 0.75344 | **2.73843** |
| NAG, clip norm 5.0 | 0.10 | 2.09610 | 0.79088 | 2.88698 |
| NAG | 0.10 | 2.13249 | 0.79131 | 2.92381 |
| NAG | 0.05 | 2.24237 | 0.78913 | 3.03150 |

Loosening the AdamW-era gradient clip from 0.5 to 5.0 improves NAG's total loss by 0.03683, but AdamW remains 0.14855 lower after equal samples. This establishes that AdamW learns much faster in the early frozen-replay regime. It does not test the claim that SGD may generalize better after a long self-play run. The selected future-run NAG schedule is a separate hypothesis: momentum 0.9, Nesterov, weight decay `1e-4`, global batch 2,048, and a linear learning-rate decay from 0.1 at generation 0 to 0.01 at generation 1,000, held thereafter.

## Evidence

Machine-readable reports are under `raw/reports/`. The retained production-size deployment artifact archive is `.codex-diagnostics/int8-scaled-post-14x160-early-fold-5k-artifacts.tar.gz` (SHA-256 `e10004af66379eb47b5f5a31c36f68082da941d05ea51dc07b71ccfc59fccdf7`). The TensorRT update-cadence and refit evidence is under `raw/cadence/`. Failed and superseded smokes are identified under `raw/failures/`; they must not be mixed with completed screen arms.
