# Bibliography and citation plan

This is the technical report's working bibliography. Primary papers and authoritative project documentation should
support algorithmic and historical claims; repository benchmarks support claims about this implementation. Entries
marked **verify** require metadata, version, and section checks during the publication pass before formal citation.

## Primary research sources

- Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
  Algorithm*. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815). AlphaZero algorithm, chess representation,
  search, optimizer, and evaluation context.
- Silver, D., Schrittwieser, J., Simonyan, K. et al. (2017). *Mastering the Game of Go without Human Knowledge*.
  *Nature* 550, 354–359. [doi:10.1038/nature24270](https://doi.org/10.1038/nature24270). AlphaGo Zero background.
- Wu, D. J. (2019). *Accelerating Self-Play Learning in Go*.
  [arXiv:1902.10565](https://arxiv.org/abs/1902.10565). KataGo fast/full searches, forced playouts, auxiliary
  targets, policy surprise, resignation, and efficiency ablations.
- Tian, Y. et al. (2019). *ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero*.
  [arXiv:1902.04522](https://arxiv.org/abs/1902.04522). Reproduction methodology, replay, search, and scaling.
- Lan, L.-C. et al. (2021). *Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search*.
  [arXiv:2012.07910](https://arxiv.org/abs/2012.07910). Adaptive stopping taxonomy and comparison.
- Czech, J., Blüml, J., Kersting, K., and Steingrimsson, H. (2023; revised 2024). *Representation Matters for
  Mastering Chess: Improved Feature Representation in AlphaZero Outperforms Switching to Transformers*.
  [arXiv:2304.14918v2](https://arxiv.org/abs/2304.14918v2). Representation and chess-architecture context; do not
  transfer its reported Elo effects to this repository without a matched experiment.
- Wu, T.-R., Guei, H., Peng, P.-C., Huang, P.-W., Wei, T. H., Shih, C.-C., and Tsai, Y.-J. (2023; revised 2024).
  *MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games*.
  [arXiv:2310.11305v3](https://arxiv.org/abs/2310.11305v3), accepted by *IEEE Transactions on Games*. Progressive
  simulation and small-board comparison context.
- Fedus, W. et al. (2020). *Revisiting Fundamentals of Experience Replay*.
  [arXiv:2007.06700](https://arxiv.org/abs/2007.06700). Replay-capacity and replay-ratio context outside AlphaZero.
- Tsai, Y.-J., Chen, W.-Y., Ju, Y.-R., Chang, Y.-H., and Wu, T.-R. (2026). *Regret-Guided Search Control for
  Efficient Learning in AlphaZero*. [arXiv:2602.20809v1](https://arxiv.org/abs/2602.20809v1), accepted at ICLR
  2026. Relevant to difficult-state selection, not evidence for this repository's exact restart-state algorithm.
- Czech, J., Korus, P., and Kersting, K. (2020). *Monte-Carlo Graph Search for AlphaZero*.
  [arXiv:2012.11045v1](https://arxiv.org/abs/2012.11045v1). Graph-search motivation; repository evidence determines
  the local rejection.

## Authoritative implementation sources

- KataGo, [`KataGoMethods.md`](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) and
  [`SelfplayTraining.md`](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md). Record a commit or
  release revision for publication rather than citing moving `master` pages.
- Leela Chess Zero, [project history](https://lczero.org/dev/wiki/project-history/). **Verify individual claims
  against dated training-run records; practitioner source, not a controlled paper.**
- Stockfish project and release documentation. Cite the exact opponent release and binary manifest used by the final
  evaluation, not only the project home page.
- NVIDIA TensorRT documentation for explicit quantization and engine refitting. **Add exact versioned pages for
  TensorRT 10.14 behavior and archive them if possible.** Repository measurements remain authoritative for the
  scale-equality failure observed here.

## Evaluation calibration

- Meloni, M. (2021). Stockfish and lc0 fixed-node strength comparison. The existing report uses this historical
  SSDF-linked calibration. **Verify the exact page, table transcription, and access date** before publication. The
  limitations in [the repository's Elo-scale analysis](../analysis/chess-elo-scale-and-reporting-20260911.md) must
  accompany any absolute benchmark Elo claim.

## Repository sources to cite as first-party evidence

- [Final run result record](../results/final-chess-run.md) — pending quantitative authority.
- [Final configuration](../../py/configs/production/chess-final-config.yaml) — living reproduction entry point.
- [v34 terminal strength](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md).
- [v34 training dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md).
- [Search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md).
- [Adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md).
- [Attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md).
- [TensorRT/INT8 feasibility](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) and
  [template failure analysis](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md).
- [Compute-poor recipe review](../analysis/reference-recipes-for-a-compute-poor-run.md), which grades external
  evidence and records sources that could not be fully verified at the time.

## Publication-pass checklist

1. Convert the working list to one consistent citation style.
2. Verify author lists, titles, years, venues, and stable URLs from primary sources.
3. Pin mutable implementation documentation to commit URLs.
4. Remove sources used only as exploratory pointers if no report claim depends on them.
5. Cite repository evidence beside each project-specific number; do not ask external papers to support local results.
6. Keep quotations minimal and verify them against the cited edition.
