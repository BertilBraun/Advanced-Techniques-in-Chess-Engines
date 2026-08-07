# AlphaZero Chess and Go Training

This project implements a full AlphaZero-style agent.  
It includes self-play data generation, presentation-credit neural network training, and evaluation.

While computationally intensive search and inference are handled in C++ (`cpp/`), Python provides one typed orchestration pipeline for chess and Go completed games, replay, self-play supervision, and optimization.

## Components

- **Self-Play Engine**  
  Generate training data by playing games against itself using the current model.

- **Training Pipeline**  
  Train the neural network on generated self-play data, with full support for multi-process parallelization.

- **Evaluation Suite**  
  Evaluate published chess models against configured baselines and previous generations.

- **UCI Engine**
  Run a published chess model in UCI-compatible chess interfaces.

## Main Scripts

- **`train.py`**  
  - Loads one required run configuration and approval record.
  - Starts persistent game-specific self-play workers and the shared trainer process.
  - Uses the same optimizer loop with no DDP wrapper when world size is one.
  - Trains optimizer quanta when replay presentation credits are available.
  - Logs training progress and metrics to TensorBoard (`logs/`).

- **`python -m src.games.chess.uci`**
  - Serves a published chess model through the UCI protocol.
  - Supports policy and retained-tree MCTS analysis modes.

## Shell Scripts

- **`train.sh`**
  - Submits a Slurm job to train the model on a compute cluster.
  - Preconfigured for reasonable resource allocation.

The training script automates distributed training job submission.

## Technologies

- **Programming Language**: Python 3.10
- **Machine Learning**: PyTorch
- **Visualization**: TensorBoard

## Workflow Overview

1. **Install Dependencies**:

    ```bash
    source setup.sh
    ```

2. **Choose** a validated presentation-credit run configuration and matching approval record.
3. **Start Training**:

    ```bash
    python train.py --run-config <approved-experiment.yaml> \
      --expected-source-revision <git-revision> \
      --approval-file <approval.json>
    ```

    The checked-in `configs/*-experiment-template.yaml` files must first be
    resolved for the selected hardware and environment.

4. **Monitor Progress**:

    Open TensorBoard:

    ```bash
    tensorboard --logdir logs/
    ```

5. **Run a Chess Model Through UCI**:

    ```bash
    python -m src.games.chess.uci --model <model.jit.pt>
    ```

6. **Optional: Submit Training to Cluster**:

    ```bash
    sbatch train.sh
    ```

## Notes

- **Self-Play**: Chess and Go use the same active-game batching and model-refresh orchestration around their concrete native search policies.
- **Persistence**: Both games use one publisher identity, atomic inbox protocol, indexed archive frame, checksum, recovery, and capacity-tail rebuild implementation.
- **Training**: One replay and trainer lifecycle materializes game-specific targets, freezes deterministic snapshots, and trains either a local rank or persistent DDP ranks.
- **Evaluation**: Existing chess evaluation remains available. Go evaluation and elapsed scheduling are intentionally deferred to R9.
