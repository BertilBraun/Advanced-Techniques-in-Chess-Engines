# Deployment and operations

- [Experiment platform](experiment-platform.md) is the current entry point for experiment configuration, fresh
  node bootstrap, approvals, queue operation, monitoring, TensorBoard, and result collection.
- [Public web play](web-play.md) covers the FastAPI/Modal backend and static browser client.
- [Lichess and Vast](lichess-vast-evaluation.md) covers the UCI/Lichess deployment and evidence collection.
- [Evaluation engines](evaluation-engines.md) covers pinned Stockfish and KataGo installation for WSL and fresh
  compute nodes.
- [R11 Vast integrated validation](vast-r11-validation.md) preserves detailed evidence from the preceding node.
- [Experiment result export](experiment-result-export.md) packages terminal queue runs, evaluation checkpoints,
  TensorBoard logs, and reproducibility metadata without copying replay data.
- [`deployment/setup_remote.sh`](../../deployment/setup_remote.sh) is the authoritative fresh training-node
  bootstrap.
- [`deployment/lichess/README.md`](../../deployment/lichess/README.md) is the executable Lichess deployment runbook.

Operational guides do not authorize external mutations, rentals, provisioning, or experiments by themselves.
