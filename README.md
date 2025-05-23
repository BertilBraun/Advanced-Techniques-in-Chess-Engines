# AlphaZero-Clone: General Deep Reinforcement Learning for Board Games

## **Project Overview**

**AlphaZero-Clone** is an open-source project implementing a **general AlphaZero-like system** for learning **board games** via **deep reinforcement learning** and **self-play**.  
Starting with simpler games like **Tic-Tac-Toe**, **Connect Four**, **Checkers** and **Hex** it is designed to scale to more complex games, such as **Chess**.

Key features:
- Learns entirely from **self-play**, with no prior knowledge of the games.
- Utilizes **Monte Carlo Tree Search (MCTS)** guided by a neural network predicting **move probabilities** (policy) and **outcomes** (value).
- Built for **scalability** across multiple GPUs and nodes.
- Extensible to multiple board games with minimal modification.
- Aims to achieve **strong amateur-level play** using **limited computational resources**.

---

## **Key Objectives**

- Implement a **general self-play training system** using MCTS and neural networks.
- Replicate AlphaZero’s success at a **strong amateur level**, **without requiring world-class hardware**.
- Start with simple board games to verify correctness, then **scale up** to more complex games like **Chess**.
- Optimize for **faster training convergence**, **efficient resource usage**, and **scalability** across multiple workers and GPUs.
- Maintain **modularity** and **readability** for easy extension to new games and research experiments.

---

## **Current Status**

### **Training System**

- The system has been **successfully validated** on **Tic-Tac-Toe** and **Connect Four** - problems started araising with **Checkers** and **Hex** - and the system doesn't seem to improve over very basic play in **Chess**.
- **Python** is used for game development and orchestration; **C++** for performance-critical components like self-play and inference.
- **Highly parallelized self-play** with **hundreds of workers**.
- **Caching mechanisms** are heavily utilized for efficiency.
- **Bayesian hyperparameter tuning** has been applied successfully to small games.

### **Critical Challenges**

> **One serious learning problem is currently blocking progress:**
> 
> - **Model stagnation**: The model stops improving after reaching ~70% win rate against random players and when started from a strong pretrained model, it deteriorates to approximately the same level.

**Fixing this issue is essential**.

---

## **Main Challenge: Model Stagnation**

> **Model stagnation is the critical project-threatening issue.**

- After reaching around **70% win rate against random opponents**, the model stops improving further.
- Continued self-play and training fail to meaningfully strengthen the model.

**Potential causes under investigation:**
- **Hyperparameter mismatches** (e.g., exploration vs. exploitation balance, learning rates).
- **Incorrect training target generation** or **evaluation logic bugs**.
- **Insufficient exploration** during self-play leading to shallow learning.
- **Training pipeline flaws** such as gradient issues, loss imbalance, or poor replay data diversity.

> **Solving model stagnation is the highest current priority for the project’s success.**

---

### **Training Logs for Debugging**

To assist in diagnosing the critical learning challenges, **detailed TensorBoard logs** and **screenshots of several training runs** are provided under: [`documentation/tensorboard_runs/`](documentation/tensorboard_runs/).

![Latest Run](documentation/tensorboard_runs/From_pretrained_py_6.png)

> **<span style="color:red">If you are experienced with reinforcement learning, deep learning training dynamics, or AlphaZero-style systems, you may be able to spot anomalies or common issues just by analyzing these plots and run documentation below — without needing to set up or run the system yourself. Any insights or help based on the provided results would be *greatly appreciated*!</span>**

<details>
<summary><strong>Training Parameters and Run documentation <span style="color:red">(click to expand)</span></strong></summary>

Pretraining was done on 1xA10 GPU for ~20min on a dataset of ~600k 2000+ Elo games. As a decent baseline, the model wins 100% of the time against random players and has baselines for policy and value predictions:
    Policy accuracy @1: 36.22%
    Policy accuracy @5: 65.42%
    Policy accuracy @10: 73.82%
    Avg value loss: 0.843603523572286

This model was then used as a starting point for the RL training run as well as for the evaluation reference model. Training then proceeded for 9 iterations with the following parameters:

```jsonc
{
  "num_gpus": 1, // 1 A10 24GB GPU
  "cpu_count": 30,
  "num_self_play_players": 27, // Based on CPU count - limited by CPU cores
  "num_games_per_iteration": 1728, // 8 (parallel games) * 27 (self players) * 8
  "num_iterations": 12,

  // Learning rate schedule (SGD)
  "learning_rate": {
    "iteration 0-9": 0.03,
    "iteration 10+": 0.01
  },

  // Network configuration
  "network": {
    "num_layers": 15,
    "hidden_size": 128
  },

  // Training settings
  "training": {
    "num_epochs": 1,
    "optimizer": "sgd",
    "batch_size": 512
  },

  // Evaluation settings
  "evaluation": {
    "num_searches_per_turn": 60,
    "num_games": 40,
    "every_n_iterations": 1,
    "dataset_path": "reference/memory_0_chess_database.hdf5"
  },

  // Self-play parameters
  "self_play": {
    "num_parallel_games": 8,
    "num_moves_after_which_to_play_greedy": 24,
    "result_score_weight": 0.15,
    "resignation_threshold": -1.0,
    "temperature": 1.0,
    "num_games_after_which_to_write": 1,

    "mcts": {
      "num_searches_per_turn": 640,
      "num_parallel_searches": 4,
      "dirichlet_epsilon": 0.25,
      "dirichlet_alpha": 0.3,
      "c_param": 1.7,
      "full_search_probability": 0.2
    }
  }
}
```

The model deteriorated rather quickly... Even though the train and validation losses decreased a lot, the evaluation shows detremental performance. The policy accuracy continuously decreased and the game win rate against the reference model dropped to a 4/19/17 win/draw/loss rate from an initial 12/17/11 win/draw/loss rate.

Models as well as additional logs are available in the [`documentation/tensorboard_runs/`](documentation/tensorboard_runs/) folder.

</details>

---

## **Training Pipeline**

The training pipeline consists of:

1. **Self-Play Generation**:
   - Multiple workers generate games using the current model and MCTS.
2. **Data Collection**:
   - Store game states, move probabilities, and outcomes to file from the workers.
3. **Training**:
   - Load the collected data from the files, shuffle and deduplicate it.
   - Train the neural network to predict policies and values from self-play data.
4. **Evaluation**:
   - Compare new models to previous best models.
   - Promote stronger models for further training.
5. **Repeat**

---

## **Optimizations**

Significant optimizations have been implemented to enhance performance, efficiency, and scalability:

- [Inference Optimization](documentation/optimizations/inference.md)
- [MCTS Optimization](documentation/optimizations/mcts.md)
- [Game Optimization](documentation/optimizations/games.md)
- [Inference Architecture Optimization](documentation/optimizations/architecture.md)
- [Training Optimization](documentation/optimizations/training.md)
- [Evaluation Optimization](documentation/optimizations/evaluation.md)
- [Hyperparameter Optimization](documentation/optimizations/hyperparameters.md)
- [Optional Pretraining](documentation/optimizations/pretraining.md) using grandmaster games and Stockfish evaluations.

---

## **Implementation Details**

See full implementation description here:  
- [Implementation Details](documentation/implementation/implementation.md)

Topics covered include:
- Neural network architecture.
- MCTS search tree structure.
- Game interface design.
- Training and evaluation loops.
- Python–C++ interoperability.

---

## **Supported Games**

- **Tic-Tac-Toe** — for basic testing and verification.
- **Connect Four** — introduces more complexity.
- **Checkers** — larger board, more complex strategies.
- **Chess** — primary target for full-scale AlphaZero-style learning.

---

## **Additional Challenges**

Besides model stagnation, there are several technical challenges:

### **1. High Cache Hit Rate**

- We currently have a **cache hit rate of 60–80%** during self-play. This means that **60–80% of the states** were already evaluated and cached.
- This could indicate duplicate evaluations or issues in the MCTS tree rebuilding or in self-play worker coordination.
- **Possible solutions** include:
  - Investigating the MCTS tree rebuilding process.
  - Ensuring that self-play workers are not duplicating efforts unnecessarily.
  - Optimizing the caching mechanism to reduce redundancy.

### **2. Hyperparameter Tuning at Scale**

- **Bayesian hyperparameter optimization** was effective for smaller games like Tic-Tac-Toe and Connect Four.
- For **Chess and complex games**, tuning becomes **very expensive** and **slow** due to long training cycles.
- Smarter, faster methods (or better initial heuristics) are needed to scale tuning efforts.

---

## **Future Work**

Future enhancements are planned:

- [Future Work Overview](documentation/future.md)

Major goals include:
- **Solve model stagnation** — **highest priority**.
- Develop **more efficient hyperparameter search** with faster hyperparameter sample evaluation.
- Implement additional **training improvements** for stability and faster convergence.

---

## **Getting Started**

The project is organized into two main sections:

- **Python Folder (py/)**:
  - Game development.
  - Training orchestration.
  - Easy-to-debug experimental framework.
- **C++ Folders (cpp/ and cpp_py/)**:
  - High-performance self-play engine.
  - Optimized multithreaded inference.

For setup and usage instructions, see the **Getting Started** guides inside each directory.

The entire project can be setup and run by launching the `getting_started.sh` script in the root directory. This will install all dependencies and set up the environment for both Python and C++ components and start the training process from the python implementation. To do this, run the following command in your terminal:

```bash
curl https://raw.githubusercontent.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/refs/heads/master/getting_started.sh | bash
```

---

## **Contributing**

Contributions are **highly welcome**, especially for:

- Fixing model stagnation.
- Debugging or improving MCTS and training.
- Optimizing C++ self-play and GPU utilization.

Please open issues or pull requests or contact the Author directly if you'd like to help!

---

## **References**

Check out [references.md](documentation/references.md) for a list of papers, articles, and resources related to AlphaZero and deep reinforcement learning.

---

## **License**

This project is licensed under the [MIT License](./LICENSE).

---

## **Acknowledgements**

- Inspired by [DeepMind's AlphaZero](https://deepmind.com/research/case-studies/alphazero-the-story-so-far).
- Utilizes [PyTorch](https://pytorch.org/) for deep learning components.

---

> **Help on fixing model stagnation is urgently needed —  
> if you have experience with AlphaZero, deep reinforcement learning, or debugging training systems, please reach out!**
