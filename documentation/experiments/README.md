# Experiment ledger

This directory turns the project's scattered benchmark records into a claim ledger. It is not a replacement for
the raw evidence. Every conclusion links to the benchmark, analysis, architecture, or plan that supports it.

The current recipe is [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). A feature's
presence there proves that it is part of the final run; it does **not** by itself prove that the feature caused a
strength gain. Conversely, historical plans show intent, not execution. The pages in this directory distinguish
those kinds of evidence deliberately.

## Status vocabulary

Every technique receives exactly one status in the topic ledgers.

| Status | Meaning |
| --- | --- |
| **Retained** | Present in the final chess recipe or an authoritative current path. The evidence column says whether retention is supported by a controlled result, a proxy, or engineering judgment. |
| **Implemented and rejected** | A working implementation was measured and the project decided not to use it. |
| **Inconclusive** | An experiment ran, but its scope, controls, statistical power, or transfer to production does not support a firm decision. |
| **Audited and declined** | The opportunity was measured or the design was examined, but full production implementation was declined. |
| **Infrastructure only** | Machinery or a smoke test was completed without an efficacy result. |
| **Proposed only** | The repository records an idea, not an implemented experiment. |
| **Superseded** | The result was once operationally relevant but a later design or measurement replaced it. |

“Retained” is therefore a recipe status, not a synonym for “validated by an isolated chess-Elo ablation.” Where the
project adopted a bundle, the ledger says so instead of assigning causal credit to every member.

## Topic narratives

- [Search](search.md): fixed and adaptive budgets, stopping, fast/full search, parallelism, tree reuse, FPU,
  forced playouts, graph search, and inference caching.
- [Data and replay](data-and-replay.md): replay reuse and capacity, policy-surprise sampling, restart states,
  resignation, cut games, weighting, reanalysis, and asynchronous operation.
- [Networks and training](networks-and-training.md): CNN/attention studies, policy heads, global context,
  progressive sizing, auxiliary heads, optimizers, learning rates, bootstrap initialization, and distillation.
- [Inference and throughput](inference-and-throughput.md): the native port, batching, concurrency, TorchScript,
  CUDA graphs, TensorRT, QAT, INT8, refit, and template lifecycle.
- [Benchmark coverage](benchmark-coverage.md): one row for every `README.md` beneath `documentation/benchmarks`,
  including nested control/failure records.

## Evidence rules

1. A benchmark result is reported as a measurement only under its recorded hardware, model, workload, and
   concurrency. Projected end-to-end gains remain projections.
2. Frozen-replay loss and policy agreement are screening proxies, not playing strength.
3. Throughput does not imply learning efficiency. In particular, work overlapped with training may not shorten the
   critical path.
4. Independent short runs are noisy. The adaptive-search fork showed why byte-identical starting state and paired
   evaluation are preferable ([conclusion](../analysis/adaptive-search-conclusion-20260904.md)).
5. Plans are evidence of a decision or intended protocol only. They are never cited as proof that a trial ran.
6. Historical measurements that a later audit invalidated remain visible and are marked superseded rather than
   silently removed.

## High-level outcome

The final recipe is best understood as an evidence-informed bundle rather than the winner of a complete factorial
experiment. The strongest conclusions are the negative ones with direct controls: learned adaptive budgets lost
strength; learned early stopping did not improve strength; graph search cost throughput for negligible reuse;
unconstrained INT8 graphs failed fidelity; and several attention comparisons were invalidated or bounded by runtime
and bootstrap defects. The strongest systems conclusions are that native batched search, appropriate concurrency,
TensorRT QAT inference, and progressive model sizes materially increase usable throughput.

The final run will determine the bundle's terminal strength. It will not retroactively turn every retained component
into an isolated causal result.
