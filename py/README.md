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
  Evaluate model performance via bot tournaments, human interaction, or matches against Stockfish.

## Main Scripts

- **`train.py`**  
  - Loads one required run configuration and approval record.
  - Starts persistent game-specific self-play workers and the shared trainer process.
  - Uses the same optimizer loop with no DDP wrapper when world size is one.
  - Trains optimizer quanta when replay presentation credits are available.
  - Logs training progress and metrics to TensorBoard (`logs/`).

- **`eval.py`**  
  - Evaluates trained models.
  - Supports:
    - **Bot vs. Bot** tournaments
    - **Human vs. Bot** interactive play
    - **Bot vs. Stockfish** matches

## Shell Scripts

- **`train.sh`**
  - Submits a Slurm job to train the model on a compute cluster.
  - Preconfigured for reasonable resource allocation.

- **`dataset_train.sh`**
  - Submits a Slurm job focused on pre-training from external datasets (e.g., grandmaster games).

These scripts automate distributed training job submissions.

## Technologies

- **Programming Language**: Python 3.10
- **Machine Learning**: PyTorch
- **Visualization**: TensorBoard

Optional dependencies for evaluation:

- `stockfish` (for engine evaluation)

## Workflow Overview

1. **Install Dependencies**:

    ```bash
    source setup.sh
    ```

2. **Choose** a validated presentation-credit run configuration and matching approval record.
3. **Start Training**:

    ```bash
    python train.py --run-config configs/chess-default-experiment.yaml \
      --expected-source-revision <git-revision> \
      --approval-file <approval.json>
    ```

4. **Monitor Progress**:

    Open TensorBoard:

    ```bash
    tensorboard --logdir logs/
    ```

5. **Evaluate Models**:

    ```bash
    python eval.py
    ```

6. **Optional: Submit Training to Cluster**:

    ```bash
    sbatch train.sh
    # or
    sbatch dataset_train.sh
    ```

## Notes

- **Self-Play**: Chess and Go use the same active-game batching and model-refresh orchestration around their concrete native search policies.
- **Persistence**: Both games use one publisher identity, atomic inbox protocol, indexed archive frame, checksum, recovery, and capacity-tail rebuild implementation.
- **Training**: One replay and trainer lifecycle materializes game-specific targets, freezes deterministic snapshots, and trains either a local rank or persistent DDP ranks.
- **Evaluation**: Existing chess evaluation remains available. Go evaluation and elapsed scheduling are intentionally deferred to R9.
