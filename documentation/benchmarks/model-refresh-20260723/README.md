# In-place model refresh benchmark

This benchmark validates the Stage 3 transactional model-refresh path at source
revision `d06aedf56539f9c8817e45085044ca0584484dd2`. It used the v5 checkpoint 319
TorchScript model without modifying or restarting the stopped production run.

The benchmark populated 24 retained search trees with 64 searches each, performed
two warm-up refreshes and ten measured refreshes, and checked root visits, live arena
nodes, and child-record counts before and after every refresh. All three inference
paths retained their trees exactly. The direct path used two inference workers with
batch size 64 and one outstanding batch.

Release/O3 was built with timing instrumentation disabled. All 11 native CTests passed
before the benchmark. The compute node used an NVIDIA GeForce RTX 4070 SUPER with
12,282 MiB VRAM and driver 595.71.05.

The production direct path refreshed in 192.2 ms on average, with a 215.6 ms p95.
Its steady RSS changed by 0.07 MiB and its GPU allocation did not grow across ten
measured swaps. The queued non-caching and caching paths refreshed in 92.2 ms and
93.6 ms on average. The direct path holds two worker-local model instances, explaining
its higher steady GPU allocation and roughly doubled refresh time.

The sampler ran every 10 ms while `refresh_model` released the Python GIL. It found no
additional observable transient GPU peak beyond the steady allocation. Raw per-refresh
JSON remains on the compute node under
`/workspace/chess/artifacts/model-refresh-d06aedf/`; aggregate results are recorded in
[results.json](results.json).

