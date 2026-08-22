# Legacy 2024 chess run (pre-rework, 4×A10)

The original trained model of the pre-rework era: commit `16a3a5b`, ~12 h on 4×NVIDIA A10
(96× Xeon Gold 6342, 256 GB RAM, $1.108/h). Full training arguments in [metadata.txt](metadata.txt);
example games vs Stockfish at Elo 2000/2200 alongside. The era's source is tagged
[`pre-rework`](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/releases/tag/pre-rework).

The large artifacts were removed from the tree and belong to the `pre-rework` GitHub release
(recoverable from git history at `documentation/benchmarks/chess-results/` before this commit):

| File | Size | SHA-256 |
| --- | ---: | --- |
| `best_model.pt` | 7,378,053 | `5572d710eefbd0bcd31044e28b1c13717319d4a2a8d6887974ba40b863753f3b` |
| `best_model.jit.pt` | 7,365,512 | `636e8ecdfc05e0bce9868c179612ff44fc4ee58c016c7ca23e85f6bac3ac381d` |
| `Example Game.gif` | 5,701,197 | `71a5dea8b7c3ef78a86baf57aae3bd560709e591385c745a0f6dbc7e98c010f2` |
| `Training Plots.png` | 1,168,827 | `73dd409e5613f83167e9711d3d559c616ca315fafbe330e11a6f17cf5b33f5bf` |
| `Training Logs.txt` | 317,231 | `e4ec3c18e6ee63390e24a6cea1f6243fbadaec5283bf78d8c107a90ca8d19aeb` |
