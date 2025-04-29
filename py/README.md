# AlphaZero-Style Chess Bot

This project implements a full AlphaZero-style chess agent entirely in Python.  
It includes self-play data generation, neural network training, evaluation, and Bayesian hyperparameter optimization.

## Components

- **Self-Play Engine**  
  Generate training data by playing games against itself using the current model.

- **Training Pipeline**  
  Train the neural network on generated self-play data, with full support for multi-process parallelization.

- **Evaluation Suite**  
  Evaluate model performance via bot tournaments, human interaction, or matches against Stockfish.

- **Hyperparameter Optimization**  
  Optimize training parameters automatically via Bayesian optimization.

## Main Scripts

- **`train.py`**  
  - Starts self-play and training according to settings defined in `src/settings.py`.
  - Handles multi-node, multi-worker training.
  - Logs training progress and metrics to TensorBoard (`logs/`).

- **`eval.py`**  
  - Evaluates trained models.
  - Supports:
    - **Bot vs. Bot** tournaments
    - **Human vs. Bot** interactive play
    - **Bot vs. Stockfish** matches

- **`opt.py`**  
  - Performs Bayesian hyperparameter optimization of all relevant training parameters defined in `src/settings.py`.
  - Iteratively searches for optimal hyperparameters (e.g., learning rate, exploration noise).

## Shell Scripts

- **`train.sh`**
  - Submits a Slurm job to train the model on a compute cluster.
  - Preconfigured for reasonable resource allocation.

- **`dataset_train.sh`**
  - Submits a Slurm job focused on pre-training from external datasets (e.g., grandmaster games).

These scripts automate distributed training job submissions.

## Technologies

- **Programming Language**: Python 3.11+
- **Machine Learning**: PyTorch
- **Optimization**: Bayesian Optimization (e.g., `scikit-optimize`)
- **Cluster Job Management**: Slurm
- **Visualization**: TensorBoard

Optional dependencies for evaluation:
- `stockfish` (for engine evaluation)

## Workflow Overview

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Configure** your training parameters in `src/settings.py`.
3. **Start Training**:

    ```bash
    python train.py
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

6. **Hyperparameter Optimization**:

    ```bash
    python opt.py
    ```

7. **Optional: Submit Training to Cluster**:

    ```bash
    sbatch train.sh
    # or
    sbatch dataset_train.sh
    ```

## Notes

- **Self-Play**: Games are generated using a Python-based implementation and saved as training datasets.
- **Training**: Trains models on collected self-play data using configurable architectures and optimizers.
- **Evaluation**: Modular system for competitive evaluations and performance tracking.
- **Optimization**: Supports automatic hyperparameter search to improve training dynamics.
