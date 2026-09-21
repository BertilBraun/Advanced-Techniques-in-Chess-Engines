# 8. Limitations

## Compute and replication

Most decisions were made under a single-node budget. Several architecture and optimizer screens used one seed, short
horizons, or stationary replay. Their within-run comparisons can be useful while still understating initialization
variance and long-horizon interaction effects. Multi-day replication of the final recipe was not affordable.

The project improved its methodology over time. Early records predate strict configuration hashes and complete
archives; later benchmarks are stronger evidence. Historical numbers should not be silently upgraded to modern
evidence standards.

## Evaluation scale

Stockfish fixed-node calibration is protocol-specific. It supports comparisons within the documented ladder but does
not produce a FIDE rating or a directly portable claim against humans, online pools, CCRL, current Stockfish, or
different hardware/time controls. “Superhuman” is supported by the chosen calibration and match conditions; it is not
a claim of grandmaster-equivalent tournament performance under every setting.

The evaluation suite uses a finite opening set and paired colors. A large match reduces sampling error but cannot
cover the full chess distribution. Search latency measurements are generally saturated batched throughput, not the
response time of one interactive game.

## Coupled interventions

The final recipe combines model growth, QAT, optimizer changes, replay changes, restart states, resignation,
auxiliary targets, and search heuristics. Not every component received a full online ablation, and interactions may
matter more than isolated effects. The report distinguishes controlled findings from bundled recipe choices.

## Proxy metrics

Policy cross-entropy, top-action agreement, KL divergence, and fixed-position accuracy are diagnostic, not strength.
The adaptive-budget work is direct evidence that a proxy can improve while Elo worsens. TensorRT work similarly
showed that a fidelity probe over random inputs and illegal actions can obscure a broken serving model.

## Search conclusions are regime-specific

Graph search was unattractive at the tested chess budgets and exact history semantics; it may differ for other games,
larger searches, or alternative repetition handling. Adaptive search failed in this overlapping training topology;
it may matter where self-play is the critical path. Attention was slower and not sufficiently better under tested
parameter and hardware constraints; this does not establish a universal CNN advantage.

## Incomplete and unattempted work

- Reanalysis was removed with an older replay design and was not tested as a modern controlled intervention.
- Fully asynchronous learner updates were not implemented; concurrency exists around explicit training quanta.
- Auxiliary-head contributions were not individually measured at full-run scale.
- Restart-state selection and policy-surprise replay were not isolated from the rest of the final bundle.
- The final run's terminal archive and evaluation are pending at this draft's cutoff.
- Repository and model licensing must be explicit before redistribution claims are made.

## Reproducibility boundary

The exact result depends on proprietary GPU drivers, CUDA/cuDNN, TensorRT behavior, rented hardware, and external
engine binaries. Locks and hashes identify these dependencies but cannot guarantee bitwise determinism across future
stacks. Some CUDA training operations are nondeterministic. Reproduction should target protocol and statistical
agreement, not necessarily byte-identical weights.
