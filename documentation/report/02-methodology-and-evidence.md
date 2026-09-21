# 2. Methodology and evidence

## Units of evidence

The repository deliberately separates evidence by purpose:

- a **configuration** states what should run;
- a **resolved configuration and hash** state what a particular run was asked to run;
- a **run archive** records what actually happened;
- a **benchmark report** interprets one bounded measurement;
- an **analysis** combines several measurements or audits a mechanism;
- a **plan** records intent and is not evidence that an experiment happened.

The benchmark index is [`documentation/benchmarks/README.md`](../benchmarks/README.md), and frozen run/node records
are indexed under [`documentation/evidence/`](../evidence/README.md). Historical plans remain useful for rejected
designs but must not be cited as completed measurements unless they point to preserved results.

## Claim states

This report uses the following vocabulary.

| State | Meaning |
| --- | --- |
| Retained | Implemented and present in the final recipe |
| Implemented, rejected | Built and measured, then removed or disabled |
| Measured, inconclusive | Evidence exists but does not support a stable decision |
| Audited, declined | Investigated far enough to reject before production adoption |
| Infrastructure only | Supporting machinery was built, but the research claim was not established |
| Proposed, not attempted | Appears in a backlog or plan only |
| Superseded | Once authoritative, replaced by a later design or result |

These states prevent a common error in long experimental projects: confusing the existence of code or a design
document with empirical support.

## Comparison hierarchy

Evidence strength increases through four levels:

1. static reasoning or literature transfer;
2. isolated throughput, fidelity, or frozen-replay measurement;
3. controlled online comparison, preferably forked from byte-identical model and replay state;
4. terminal match evidence with fixed opponents, balanced openings, complete artifacts, and uncertainty intervals.

The adaptive-stopping campaign showed why shared-state forks matter: independent runs had much greater ladder noise
than arms resumed from the same checkpoint and replay state. See the
[adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md). Frozen-replay studies are excellent
for optimizer, architecture, and quantization diagnosis, but they do not by themselves establish self-play Elo.

## Evidence dimensions

A single “strong/weak” label is too imprecise for this project. Conclusions are classified by the observation they
actually contain:

| Grade | Observation | Valid use |
| --- | --- | --- |
| **S — strength** | Paired games or a calibrated ladder under a frozen protocol | Strength claim for that checkpoint, search, and opponent |
| **O — online learning** | Self-play learning slope, preferably from a shared-state fork | Learning-system comparison within the measured regime |
| **P — proxy** | Frozen replay, held-out loss, target fidelity, policy agreement, or fixed-batch fit | Candidate selection or mechanism diagnosis, not Elo |
| **T — throughput** | Forward, search, actor, trainer, or admitted-replay rate | Performance claim under the recorded workload, not learning quality |
| **M — mechanics** | Unit/integration test, smoke, persistence audit, or telemetry | Correctness and operation, not efficacy |
| **R — rationale** | Literature transfer, design analysis, or an unexecuted plan | Motivation only |

A technique can have several grades. Progressive sizing has **T** evidence for the small-model premise and **M**
evidence for durable promotion, but no isolated **O/S** comparison of the exact final ladder against a fixed model.
QAT has **P/T/M** evidence; the final run determines whether the resulting extra data production converted to
learning and strength.

The [experiment ledger](../experiments/README.md) records technique status, the
[benchmark coverage ledger](../experiments/benchmark-coverage.md) accounts for benchmark artifacts, and the
[report coverage matrix](coverage-matrix.md) records where every analysis and benchmark family enters this report.

## Corrections and supersession

The repository preserves intermediate interpretations when they explain how a diagnosis changed. This report uses
the latest controlled conclusion and states the correction. Examples include FP32 attention measurements that did
not represent production BF16, a parallel-search sweep whose batch regime did not exercise the intended cap, and
the initial “template staleness” explanation superseded by the TensorRT equal-scale refit defect.

Where raw evidence was not preserved—most notably an early low-rank dense policy-head bake-off—the report may
describe the historical direction but must not promote exact numbers to the same status as a tracked benchmark.

## Strength measurement

Chess strength is measured through paired-opening matches against Stockfish at fixed node limits. Reported absolute
ratings use the project's historical SSDF-derived Stockfish-node calibration. They are **benchmark Elo**, not FIDE,
online-server, CCRL, or universally portable engine ratings. Every published strength row should retain:

- checkpoint and inference-backend identity;
- searches per move and search parallelism;
- opponent version, node limit, threads, and hash;
- opening source and number of paired openings;
- W/D/L, score, sample count, confidence interval, and rating conversion;
- hardware and, when latency is claimed, concurrency and batching assumptions.

The interpretation and acceptable language are defined in
[the Elo-scale note](../analysis/chess-elo-scale-and-reporting-20260911.md). The v34 terminal protocol provides the
clearest completed example in the [generation-1465 benchmark](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md).

## Throughput and fidelity

Throughput is workload-specific. Batch size, concurrent roots, model state, process topology, CUDA graph capture,
precision, and GPU contention can change a result. A saturated many-game measurement is not single-game response
latency. Architecture comparisons should separate raw forward throughput from search throughput because different
policies can cause different numbers of network evaluations.

Inference speed is never sufficient on its own. TensorRT artifacts are checked against a reference model, and the
INT8 investigation ultimately showed why probes must use real positions and legal-action masking. The
[template-staleness investigation](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) records both
the failed metric and the corrected diagnosis.

## Evidence rules for the final run

The readable, living recipe and the frozen scientific identity serve different purposes. The living entry point is
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). The final result must additionally
freeze the exact source revision, resolved configuration and SHA-256, run manifest, hardware/runtime inventory,
selected checkpoint, inference artifacts, archive digest, and evaluation artifacts. If the living configuration is
updated later, the reported experiment must remain reproducible from its frozen identity.

All pending terminal measurements are listed once in [Chapter 7](07-final-run-results.md). Other chapters describe
methods and prior evidence without guessing the outcome.
