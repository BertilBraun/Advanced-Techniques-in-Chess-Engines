# Resignation audit compute-node canary

This artifact records the Stage 1 resignation-audit validation on the four-GPU
compute node. The implementation was built from
`3bcdeadb3cf8481dd1cba9206ad1adffac7b9596` in the isolated worktree
`/workspace/chess/resignation-canary-source`. The live v5 training checkout remained
on its pinned revision `1806a2df9de9683f5261c87b6c259f797fad0cce`.

The native extension was configured as Release/O3 with timing instrumentation
disabled and LibTorch's C++11 ABI enabled. All 11 native CTests passed. The focused
resignation, self-play, dataset, replay-buffer, and run-configuration Python suite
passed all 93 tests against the freshly built extension.

The production-path smoke test used checkpoint 304 through the non-caching inference
client on GPU 3. It ran one parallel game for 12 self-play steps with 64 full searches,
16 fast searches, and four parallel searches. A deliberately aggressive `-0.01`
cutoff and a 10-ply cap made the audit continuation and persistence paths execute
quickly without changing the live run.

The completed audit persisted four full-search observations and its first hypothetical
resignation at ply 5. Its termination reason was `ply_cap`, so the calibration correctly
excluded it from safety evidence and left production resignation disabled. The cutoff,
cap, and 100% audit assignment were test-only settings; they are not proposed training
parameters and do not establish a statistically safe production threshold.

Machine-readable results are in [results.json](results.json).
