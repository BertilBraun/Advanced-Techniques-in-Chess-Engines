# AlphaZero-Clone: General Deep Reinforcement Learning for Board Games

## **Project Overview**

**AlphaZero-Clone** is an open-source project implementing a **general AlphaZero-like system** for learning **board games** via **deep reinforcement learning** and **self-play**.  
Starting with simpler games like **Tic-Tac-Toe** and **Connect Four**, it is designed to scale to more complex games, such as **Chess**.

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

- The system has been **successfully validated** on **Tic-Tac-Toe** and **Connect Four**.
- **Python** is used for game development and orchestration; **C++** for performance-critical components like self-play and inference.
- **Highly parallelized self-play** with **hundreds of workers**.
- **Caching mechanisms** are heavily utilized for efficiency.
- **Bayesian hyperparameter tuning** has been applied successfully to small games.

### **Critical Challenges**

> **Two serious learning problems are currently blocking progress:**
> 
> - **Model stagnation**: The model stops improving after reaching ~70% win rate against random players.
> - **Early value head collapse**: The value head collapses to predicting a constant value after just one epoch of training.

**Fixing these issues is essential** to enable scaling the project to complex games like Chess.

---

## **Main Challenge: Model Stagnation**

> **Model stagnation is the critical project-threatening issue.**

- After reaching around **70% win rate against random opponents**, the model stops improving further.
- Continued self-play and training fail to meaningfully strengthen the model.
- **Without solving this**, scaling up to more complex games like **Chess** becomes impossible.

**Potential causes under investigation:**
- **Hyperparameter mismatches** (e.g., exploration vs. exploitation balance, learning rates).
- **Incorrect training target generation** or **evaluation logic bugs**.
- **Insufficient exploration** during self-play leading to shallow learning.
- **Training pipeline flaws** such as gradient issues, loss imbalance, or poor replay data diversity.

> **Solving model stagnation is the highest current priority for the project’s success.**
---

### **Training Logs for Debugging**

To assist in diagnosing the critical learning challenges, **detailed TensorBoard logs** and **screenshots of several training runs** are provided under: `documentation/tensorboard_runs/`

These logs include:

- Cache hit rates
- Dataset statistics
- Policy target distributions
- Value target distributions
- Evaluation results
- Loss curves
- GPU utilization
- Neural network outputs
- Training and validation performance over time
- Value output means and standard deviations

> **If you are experienced with reinforcement learning, deep learning training dynamics, or AlphaZero-style systems,  
> you may be able to spot anomalies or common issues just by analyzing these plots —  
> without needing to set up or run the system yourself.  
> Any insights or help based on the provided results would be *greatly appreciated*!**

---

## **Training Pipeline**

The training pipeline consists of:

1. **Self-Play Generation**:
   - Multiple workers generate games using the current model and MCTS.
2. **Data Collection**:
   - Store game states, move probabilities, and outcomes.
3. **Training**:
   - Train the neural network to predict policies and values from self-play data.
4. **Evaluation**:
   - Compare new models to previous best models.
   - Promote stronger models for further training.

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

Besides model stagnation, there are several critical technical challenges:

### **1. Early Value Head Collapse**

- The **value head** of the neural network collapses to predicting a **constant value** (e.g., always zero or always the same number) **within the first training epoch**.
- After collapse, the value output becomes **uninformative**, and the model **fails to learn** the true game outcomes.
- This behavior persists **regardless of training data** or further optimization.
- **Potential causes** may include:
  - Incorrect loss function setup or imbalance between value and policy loss.
  - Faulty target generation during self-play (e.g., wrong labeling of outcomes).
  - Poor initialization or optimization parameters.
  - Bugs in backpropagation, training loops, or MCTS result extraction.

> **Fixing value head collapse is a high priority**, as it is likely connected to the broader model stagnation problem.

### **2. High Cache hit Rate**

- We currently have a **cache hit rate of 60–80%** during self-play. This means that **60–80% of the states** were already evaluated and cached.
- This means that we have a lot of duplicate evaluations, which could be from the MCTS trees, being rebuilt every time a new move is played, or from the same board state being reached by different move sequences or problems between the self-play workers.
- This could be totally fine and simply by design of the MCTS, but it could also be a major bug/problem in the self-play engine or the MCTS implementation.
- **Possible solutions** include:
  - Investigating the MCTS tree rebuilding process.
  - Ensuring that self-play workers are not duplicating efforts unnecessarily.
  - Optimizing the caching mechanism to reduce redundancy.

### **3. GPU Utilization**

- During large-scale self-play, GPU usage remains around **60–70%**.
- Possible causes:
  - Inefficient inference batching or resource utilization.
  - Suboptimal threading or caching techniques.
- While important for speeding up training, this issue is **secondary** compared to the learning problems.

### **4. Hyperparameter Tuning at Scale**

- **Bayesian hyperparameter optimization** was effective for smaller games like Tic-Tac-Toe and Connect Four.
- For **Chess and complex games**, tuning becomes **very expensive** and **slow** due to long training cycles.
- Smarter, faster methods (or better initial heuristics) are needed to scale tuning efforts.

---

## **Future Work**

Future enhancements are planned:

- [Future Work Overview](documentation/future.md)

Major goals include:
- **Solve model stagnation and value collapse** — **highest priority**.
- Improve **C++ self-play optimization** for better GPU/CPU utilization.
- Develop **more efficient hyperparameter search** for scaling to Chess and beyond.
- Expand the system to handle other board games with varied rulesets.
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

---

## **Contributing**

Contributions are **highly welcome**, especially for:

- Fixing model stagnation and value collapse.
- Debugging or improving MCTS and training.
- Optimizing C++ self-play and GPU utilization.
- Adding support for new board games.

Please open issues or pull requests if you'd like to help!

---

## **References**

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero Paper)](https://arxiv.org/pdf/1712.01815)
- [Minigo: A Case Study in Reproducing Reinforcement Learning Research](https://openreview.net/pdf?id=H1eerhIpLV)
- [Lessons from AlphaZero: Connect Four](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68)
- [Lessons from AlphaZero Part 3: Parameter Tweaking](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)
- [Lessons from AlphaZero Part 4: Improving the Training Target](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
- [Lessons from AlphaZero Part 5: Performance Optimization](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)
- [Lessons from AlphaZero Part 6: Hyperparameter Tuning](https://medium.com/oracledevs/lessons-from-alphazero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
- [AlphaZero Chess: How It Works, What Sets It Apart, and What It Can Tell Us](https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867)
- [AlphaZero Explained](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [AlphaGo Zero Cheat Sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
- [AlphaZero from Scratch (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org)
- [AlphaZero General (GitHub)](https://github.com/suragnair/alpha-zero-general)

---

## **License**

This project is licensed under the [MIT License](./LICENSE).

---

## **Acknowledgements**

- Inspired by [DeepMind's AlphaZero](https://deepmind.com/research/case-studies/alphazero-the-story-so-far).
- Utilizes [PyTorch](https://pytorch.org/) for deep learning components.

---

> **Help on fixing model stagnation and value collapse is urgently needed —  
> if you have experience with AlphaZero, deep reinforcement learning, or debugging training systems, please reach out!**
