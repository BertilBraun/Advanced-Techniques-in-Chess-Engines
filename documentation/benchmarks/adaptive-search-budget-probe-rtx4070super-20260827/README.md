# Adaptive search budget frozen-trunk probe — RTX 4070 SUPER, 2026-08-27

## Result

The mandatory offline gate for the learned per-position search budget **failed**.

The scalar-head-shaped predictor found a real relationship between generation-162 trunk features and the
8.3×-depth KL label (out-of-fold Spearman 0.592), but its ordering did not clear the allocation convexity hole.
At the oracle's exact mean visit spend it captured **-13.95% of oracle gain** relative to a flat 600-visit budget.
The 2,000-sample bootstrap interval was entirely below flat: **[-24.49%, -3.92%]**.

Per the hard-stop rule in
[`adaptive-search-budget-20260827.md`](../../plan/adaptive-search-budget-20260827.md), the production allocator,
deep-labelling lifecycle, replay write-back, and fast/full-search replacement were not implemented.

## Inputs and provenance

- Source revision: `3cd32592f3a75448902dd14331db6b6dd5e2ff10`
- Probe tool SHA-256: `4ec2d32b73f059f4cd2d267a41620d6bcda04e3499dabbe09cc61ae6d9198468`
- Frozen v9 generation-162 model SHA-256:
  `a61e850264c71c9c4fd38f0f9660b64b6737052b0516245a34b4e79f3f098571`
- Checked-in 3,000-position depth-sweep SHA-256:
  `ae9826fce5eedd837560e37e7a4d6983bfed6cd56ae8d13f6637d910188ddf2f`
- Device: NVIDIA GeForce RTX 4070 SUPER, 12 GB
- Runtime: Python 3.12.3, PyTorch 2.12.1+cu126
- Focused validation: `5 passed` in `11.99s`

The active v9 training node was used only to copy the already frozen `model_162.jit.pt`; no training process,
service, run state, or live checkpoint was changed. Feature extraction and predictor fitting ran on the separate
4070 SUPER node.

## Equal-compute scores

| Allocation | Mean visits | Mean KL | Oracle gain captured |
|---|---:|---:|---:|
| Flat 600 | 600.000 | 0.323906 | 0.00% |
| Noise-corrected oracle | 599.933 | 0.147562 | 100.00% |
| Perfect 5,000-visit KL label | 599.933 | 0.231772 | 52.25% |
| Frozen-trunk predictor | 599.933 | 0.348507 | **-13.95%** |
| Random-order mean (100 orderings) | 599.933 | — | -87.51% |

The predictor substantially outperformed random ordering, but that is not the gate. A learned non-uniform
allocator must beat the flat budget at equal compute; this one did not.

The machine-readable result is [`probe-report.json`](probe-report.json).
