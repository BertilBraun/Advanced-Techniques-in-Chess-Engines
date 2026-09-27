# Engineering Efficient Self-Play Chess

This release publishes the technical report and the accompanying chess system, documentation, configuration, and preserved experimental evidence.

## Results

Training from random initialization through self-play on one node with eight RTX 4070 SUPER GPUs produced a 6.32-million-parameter model in 2.5 days. The reported run ingested 3.25 million completed games and made 836.6 million training presentations, at a node rental cost of $43.20 for that training budget.

The final model reached **3,251 benchmark Elo [3,206, 3,297] at 100,000 searches per move**, compared with **1,658 [1,608, 1,710] without search**. Ratings use a historical fixed-node Stockfish 13 calibration, not FIDE ratings. The approximately 470-thousand-parameter distilled student reached **2,873 [2,819, 2,935]** at 100,000 searches per move.

## What is included

- The complete technical report, attached as **engineering-efficient-self-play-chess.pdf**, with methods, results, failure studies, references, and detailed appendices.
- Native C++ search and batched inference, Python orchestration and distributed training, replay management, and evaluation tooling.
- The final chess configuration, progressive model sizing, restart-state sampling, replay policies, calibrated resignation, and TensorRT deployment.
- Benchmarks and analyses covering retained improvements and rejected approaches, including graph search, inference caching, adaptive search, alternative architectures, and quantization.
- Reworked project documentation and report-generation tools.

The reported strength belongs to the integrated system; the report distinguishes full training results from individual controlled comparisons. Later experiments do not extend the reported 2.5-day training budget. The separate LC0-teacher work remains experimental and is not included in this release.

## Use the project

- [Play against the model](https://chess.bertil-braun.de/)
- [Model artifacts and model card](https://huggingface.co/BertilBraun/alphazero-chess)
- The source archive includes `py/configs/production/chess-final-config.yaml`, the entry point for the maintained training recipe.
- See the root README for setup and `documentation/report/README.md` for report sources and evidence.

Original project code, documentation, and model artifacts are MIT-licensed; third-party dependencies retain their own licenses. This is the GitHub publication of the paper; an arXiv submission is not part of this release.
