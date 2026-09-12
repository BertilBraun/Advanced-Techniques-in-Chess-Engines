# TensorRT INT8 architecture screen

This bounded screen tested whether a quantization-friendly residual trunk could preserve the roughly
3x isolated INT8 throughput observed for the terminal v34 network while making QAT fidelity
recoverable. It did not start the planned 100,000-step factorial because the deployment graph failed
both the fidelity and practical-throughput prerequisites during smoke tests.

## Architecture and identity

The proposed model uses 14 residual blocks at width 160, global pooling in every second block, a
from-to policy head, pre-activation `BN -> ReLU6 -> Conv` branches, and residual-branch scaling of
`1/sqrt(14)`. With the v34 52-plane input it has 6,261,007 parameters and 395,894,544 MACs. The
parameter-matched 12x128 form has 3,478,151 parameters and 219,904,144 MACs; the earlier 3,451,655
count applied to the old 29-plane input.

All measurements used batch 320 on one RTX 4070 SUPER with PyTorch 2.12.1+cu126, TensorRT
10.14.1.48.post1, and ModelOpt 0.46.1. The original-network repeats used the generation-1785 v34
checkpoint (`8899d1d4fedda4dc0faf85b54c2d433dde65d5953be0ae078ea69354ec01d4bf`). Each row is the
mean of two interleaved engine rebuilds; the range is shown in parentheses.

## Repeated isolated throughput

| Graph | TorchScript BF16 positions/s | TensorRT FP16 positions/s | TensorRT INT8 positions/s | INT8 / TorchScript | INT8 / TRT FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v34 14x160 | 61,293 (61,223-61,362) | 101,464 (100,972-101,957) | 183,592 (182,134-185,050) | 2.995x (2.975-3.016x) | 1.810x (1.786-1.833x) |
| Pre-activation 14x160 | 61,032 (61,015-61,050) | 85,991 (85,987-85,994) | 114,359 (114,331-114,387) | 1.874x (1.873-1.875x) | 1.330x (1.330-1.330x) |

The nearly identical TorchScript rates resolve the denominator question: the lower TensorRT result is
caused by the pre-activation/QDQ graph, not a smaller or slower PyTorch baseline. Its FP16 graph loses
TensorRT fusion efficiency and its INT8 graph gains only another 1.33x over that lower FP16 baseline.

The original full-trunk ONNX contains 58 Q/DQ pairs for 29 convolutions. The proposed graph contains
56 pairs for the intended 28 residual convolutions. Q/DQ counts conceal the important difference:
the original block layout lets TensorRT keep most of a residual block in one quantized fused tactic,
while the pre-activation layout materializes normalization, clipping, scaling, and format changes.

The first detailed pre-activation inspector artifact was accidentally built from the original ONNX
graph. A rebuild from the recorded pre-activation ONNX identity corrected the comparison:

| Engine | Layers | Reformats | Conv + ReLU | Conv + residual + ReLU | Pointwise layers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FP16 | 83 | 1 | 8 | 14 | 0 |
| Pre-activation FP16 | 330 | 70 | 1 | 0 | 1 |
| Original INT8 | 103 | 18 | 8 | 14 | 0 |
| Pre-activation INT8 | 470 | 102 | 1 | 0 | 35 |

The original INT8 engine uses 21 trunk convolutions whose inputs and outputs remain INT8, including
all 14 second convolutions fused with the residual add and following ReLU. Seven global-pooling
convolutions produce FP32 for their pooling islands. In the pre-activation INT8 engine, 21 trunk
convolutions instead use INT8 inputs and weights but produce FP16; the engine then executes the
pre-activation batch normalization and clipping, branch-scale multiplication, residual addition,
and requantization separately. Its detailed tensor records contain 390 FP16 outputs and only 35 INT8
outputs, compared with 20 FP16 and 38 INT8 outputs in the original engine.

The same structural cost is already visible without quantization. The pre-activation FP16 ONNX adds
28 clipping operations and 14 branch-scale multiplications, and its TensorRT engine grows from 83 to
330 layers. This accounts for the FP16 throughput decline from 101,464 to 85,991 positions/s. The
additional FP16/INT8 transitions and lost residual-block fusion then limit INT8 to 1.33x over that
slower FP16 engine, rather than the original graph's 1.81x. Multiplying those two effects produces
the observed 1.874x rather than 2.995x speedup over TorchScript.

Explicitly forcing the start convolution, global-context operations, residual adds, and heads to
FP16 made TensorRT 10.14 reject the weakly typed QDQ graph during format selection. A strongly typed
q1 engine ran at only 63,203 positions/s and still failed fidelity.

The most plausible recovery is therefore a bounded, scaled **post-activation** block. It retains the
original `Conv -> BN -> activation -> Conv -> BN -> residual add -> activation` deployment order,
folds batch normalization into convolution, absorbs residual-branch scaling into the second
convolution at export, and aligns the branch and skip quantizers so TensorRT can fuse the residual
input. This keeps the activation cap and residual scaling that constrain QAT ranges without placing
normalization and clipping between quantized convolution tactics. Conv-BN-aware QAT or explicit
block-boundary fake quantization are the remaining implementation routes if ordinary module-level
QAT cannot learn this graph. Reducing global-pooling frequency could remove FP32 islands but would
change the model's information path and does not explain the measured regression.

## Fidelity isolation

ONNX Runtime reproduced the framework fake-quant graph closely from one through 24 QAT convolutions,
which rules out double quantization and loss of learned scales during export. TensorRT divergence began
with the first pre-activation branch and compounded. After 100 bounded replay steps, the pre-activation
q1 engine had 80.9% TensorRT-versus-ORT policy top-1 agreement and KL 0.0568; q24 had 53.8% and KL
0.348. By comparison, a trained post-activation q1 graph reached 98.75% and KL 0.0000757. Folding its
learned Conv/BN statistics changed little, so missing post-training BN folding was not the cause.

The random 14x160 pre-activation engine is not a chess-strength candidate and also fails the runtime
fidelity gate. The screen therefore provides no basis for the full replay-training factorial or chess
games.

## Practical ceiling

Production v34 measured 48,463 positions/s/GPU against 61,293 positions/s for isolated TorchScript.
Treating the implied 4.319 microseconds/position as fixed search and transport overhead projects the
pre-activation INT8 graph to about 76,549 positions/s/GPU, or **1.58x end-to-end**. This is an Amdahl
projection, not a production measurement. The invalid original full-trunk INT8 graph projects to
2.11x, which confirms that the 3x core result could have crossed the user's 2x practical threshold if
it had preserved model behavior.

The quantization-friendly design cannot plausibly reach the required 2x practical self-play speedup
under the tested TensorRT graph, even before accounting for its unresolved fidelity error. The
100,000-step eight-arm replay screen was therefore not launched.

Raw report SHA-256 identities are recorded in `raw/report-sha256.txt`. Large ONNX graphs, TensorRT
engines, and the 1.1 MB detailed inspector output remain on the node; the inspector hash is included in
the same manifest.
