# Python Training and Evaluation for AlphaZero Chess Bot

This Python module orchestrates training and evaluation of the AlphaZero-style chess bot using the high-performance C++ self-play engine provided via PyBind.

While computationally intensive components (e.g., self-play) are handled in C++ (`cpp/`), the Python side manages the training loops, evaluation, and cluster orchestration.

## Components

- **`train.py`**  
  Starts a full training run based on the settings defined in `cpp_py/src/settings.py`.
  - Handles multi-node, multi-worker training.
  - Self-play data is generated using the C++ backend.
  - Training progress and metrics are logged to TensorBoard (`logs/` directory).

- **`eval.py`**  
  Provides various evaluation routines:
  - **Bot vs. Bot Tournament**: Evaluate different model versions against each other.
  - **Human vs. Bot Interface**: Play interactively against the bot.
  - **Bot vs. Stockfish**: Benchmark the bot’s performance against Stockfish.

## Shell Scripts

- **`train.sh`**
  - Submits a Slurm job to train the bot on a compute cluster.
  - Configured with reasonable defaults for node and resource allocation.
  
- **`dataset_train.sh`**
  - Submits a Slurm job focused on pre-training using datasets (e.g., grandmaster games, Stockfish evaluations).

These scripts handle cluster submission and resource management.

## Dependencies

- Python 3.11+
- PyTorch
- PyTorch Lightning (or minimal custom trainer)
- TensorBoard
- [AlphaZeroCpp](../cpp_py) (the C++ self-play engine built as a Python module)

Ensure the C++ backend is built and the `AlphaZeroCpp.so` is available in the `cpp_py/` directory as well as the `py/setup.sh` script has run before starting training.

## Workflow Overview

1. **Build** the C++ shared object (`AlphaZeroCpp.so`) following the [C++ README](../cpp/README.md).
2. **Configure** your settings in `cpp_py/src/settings.py`.
3. **Start Training**:
    ```bash
    python train.py
    ```
4. **Evaluate Models**:
    ```bash
    python eval.py
    ```
5. **Optional: Submit Training Jobs to Cluster**:
    ```bash
    sbatch train.sh
    # or
    sbatch dataset_train.sh
    ```

## Notes

- Self-play workers are launched via the C++ module (`AlphaZeroCpp`) and write training data to disk.
- The training script automatically reads and batches the generated data.
- Evaluation modes are modular and can be easily extended.
