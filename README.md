# AlphaZero-Clone: General Deep Reinforcement Learning for Board Games

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BertilBraun/Advanced-Techniques-in-Chess-Engines)

> **Achievement:** Intermediate-level Chess AI (~2000-2100 Elo) trained on a $350 personal R&D budget with final model costing $13 to train.

## **Project Overview**

**AlphaZero-Clone** is a personal implementation of AlphaZero's deep reinforcement learning approach, designed to master board games through self-play without human knowledge. This project demonstrates that sophisticated AI techniques can achieve strong performance on modest personal budgets, successfully scaling from simple games like Tic-Tac-Toe to complex strategy games like Chess.

The system combines Monte Carlo Tree Search (MCTS) with deep neural networks to learn optimal play through millions of self-play games. The implementation focuses on practical optimization and efficiency, achieving intermediate-level Chess play on limited computational resources.

### **Core Features**

- **General Game Framework**: Extensible architecture supporting multiple board games with minimal modification
- **Self-Play Learning**: Learns optimal strategies without human knowledge or game-specific heuristics
- **Distributed Training**: Scalable across multiple GPUs and CPU cores with asynchronous data generation
- **Production-Ready Performance**: Optimized C++ MCTS implementation achieving 4-10k searches per second
- **Budget Efficient**: Achieves strong performance with limited computational resources

---

## **Supported Games and Results**

### **Game Progression**

| Game             | Board Size   | Complexity   | Status             | Performance Notes                  |
| ---------------- | ------------ | ------------ | ------------------ | ---------------------------------- |
| **Tic-Tac-Toe**  | 3x3          | ~10³ states  | Solved             | Perfect play achieved              |
| **Connect Four** | 7x6          | ~10¹³ states | Mastered           | Strong tactical understanding      |
| **Checkers**     | 8x8          | ~10²⁰ states | Strong             | Solid positional and tactical play |
| **Hex**          | 7x7 to 11x11 | ~10²⁵ states | Strong             | Effective on multiple board sizes  |
| **Chess**        | 8x8          | ~10⁴⁷ states | **~2000-2100 Elo** | **Intermediate-level mastery**     |

### **Chess Performance Analysis**

The Chess implementation represents the project's main achievement, demonstrating sophisticated understanding of both tactical and positional concepts. Training utilized approximately 280,000 games over 100,000 update steps with a compact 8x96 neural network architecture.

![Sample Game vs Stockfish Level 6](documentation/chess_results/Example%20Game.gif)
*AlphaZero-Clone (White) defeating Stockfish Level 6 in a tactical middlegame*

#### **Benchmark Results**

Performance was evaluated against multiple opponents across various time controls:

**Stockfish UCI Level Testing:**

| Stockfish Level | Win/Draw/Loss |
| --------------- | ------------- |
| Level 0         | 8/1/1         |
| Level 3         | 7/1/2         |
| Level 4         | 53/17/30      |
| Level 5         | 7/6/7         |
| Level 6         | 2/6/12        |

**Elo-Calibrated Testing:**

| Target Elo | Result |
| ---------- | ------ |
| 1400       | 20/0/0 |
| 2000       | 12/4/4 |
| 2200       | 7/4/9  |
| 2400       | 5/7/8  |

*The engine also passed a more personal benchmark: winning a casual game against my Dad ❤️.*

*Testing conditions: 1.0s/move time control, Stockfish skill level limited via `Skill Level`, Elo targets set via `UCI_Elo`*  
*Testing hardware: Single A10 GPU + 16 CPUs vs Stockfish on 32 CPUs*

The performance data suggests the engine operates at approximately **2000-2100 Elo**, demonstrating intermediate-level play with solid tactical awareness and positional understanding.

---

## **Technical Architecture**

The system is built around a robust training pipeline that separates concerns between data generation, neural network training, and model evaluation:

### **Training Pipeline**

1. **Self-Play Generation**: Multiple worker processes generate games using the current best model and MCTS
2. **Data Collection**: Game states, MCTS-derived move probabilities, and final outcomes are stored efficiently
3. **Neural Network Training**: Collected data trains the network to predict both move probabilities (policy) and position evaluations (value)
4. **Model Evaluation**: New models are benchmarked against previous versions to ensure improvement
5. **Model Promotion**: Superior models become the new best model for continued self-play generation

### **Implementation Details**

The project leverages a hybrid Python/C++ architecture optimized for both development velocity and runtime performance:

- **Python Components**: Training orchestration, neural network definitions, experimental framework
- **C++ Components**: High-performance MCTS implementation, game engines, batched inference
- **Asynchronous Design**: Self-play workers operate independently while training occurs on accumulated data
- **Mixed Precision**: BFloat16 training and inference for improved performance and reduced memory usage

---

## **Training Methodology**

### **Resource Efficiency**

The project achieves remarkable results through careful resource management:

- **Total Personal Budget**: ~$350 including all experimental runs
- **Final Model Training Cost**: ~$13 for 12-hour training session
- **Hardware Configuration**: 96 CPU cores + 3.5 A10 GPUs
- **Training Throughput**: 5-15 games per second during self-play generation

### **Implementation Techniques**

The implementation incorporates several established techniques for improved performance:

- **AdamW Optimizer**: Superior convergence properties for deep learning
- **Mixed Precision Training**: BFloat16 reduces memory usage while maintaining numerical stability
- **ResNet Architecture**: Skip connections with Squeeze-and-Excitation blocks for efficient learning
- **Playout-Cap Randomization**: Technique borrowed from KataGo for improved exploration
- **Asynchronous Training**: Continuous data generation while neural network training occurs

<details>
<summary><strong>Training Progression Analysis</strong></summary>

The training process demonstrates clear learning phases visible in comprehensive metrics:

![Training Plots](documentation/chess_results/Training%20Plots.png)

**Key Observations:**

- **Loss Convergence**: Policy and value losses show steady improvement with occasional plateaus
- **Value Accuracy**: Position evaluation accuracy improves consistently throughout training
- **Search Efficiency**: MCTS becomes more selective as the neural network improves
- **Evaluation Metrics**: Win rates against previous models and Stockfish show clear upward trends

The training data reveals the model's learning trajectory from random play to sophisticated chess understanding, with the skill level increasing significantly over time. Notably, the model does not seem to have fully converged yet, indicating potential for further improvement with additional training resources. Additionally, increases in model size, training duration and number of games played will most likely lead to even stronger performance.

</details>

---

## **Performance Optimizations**

The training pipeline includes several important optimizations for efficiency:

### **Core Optimizations**

- **[Inference Optimization](documentation/optimizations/inference.md)**: Batched GPU inference with efficient memory management
- **[MCTS Optimization](documentation/optimizations/mcts.md)**: Optimized tree traversal and node expansion algorithms
- **[Game Engine Optimization](documentation/optimizations/games.md)**: Efficient board representation and move generation
- **[Training Pipeline Optimization](documentation/optimizations/training.md)**: Data loading, shuffling, and batch processing improvements
- **[Evaluation Optimization](documentation/optimizations/evaluation.md)**: Parallel game evaluation and statistical analysis

### **System Design**

- **Asynchronous Architecture**: Eliminates training bottlenecks through independent worker processes
- **Memory Management**: Efficient data structures minimizing memory allocation overhead
- **Batch Processing**: Optimal batch sizes for both training and inference operations
- **Load Balancing**: Dynamic work distribution across available computational resources

---

## **Potential Improvements**

The training plots suggest the model hasn't fully converged yet, indicating significant room for improvement with additional computational resources:

- **Extended Training**: Performance likely to improve with longer training runs and more games
- **Larger Networks**: Scaling to bigger neural network architectures
- **Game Expansion**: Support for Go, Shogi, and other complex strategy games
- **Training Techniques**: Population-based training and advanced exploration methods
- **Hyperparameter Optimization**: Systematic search for optimal training configurations

Detailed roadmap available in **[Future Work](documentation/future.md)**.

---

## **Getting Started**

### **Quick Start**

The entire system can be installed and configured with a single command:

```bash
curl https://raw.githubusercontent.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/refs/heads/master/getting_started.sh | bash
```

This script handles dependency installation, environment configuration, and initiates the training process.

### **Project Structure**

- **`py/`**: Python components for training orchestration and experimentation
- **`cpp/`**: High-performance C++ implementations for self-play and inference
- **`documentation/`**: Comprehensive technical documentation and analysis
- **`documentation/chess_results/`**: Detailed Chess performance data and sample games

For detailed setup instructions and usage examples, refer to the Getting Started guides in each directory.

---

## **Contributing**

Contributions are welcomed across multiple areas:

- **Algorithm Implementation**: Training techniques and search improvements
- **Game Support**: Additional board games and rule variants
- **Performance Optimization**: Speed and memory efficiency improvements
- **Documentation**: Technical documentation and educational resources
- **Evaluation**: Benchmarking and performance analysis tools

Please review contribution guidelines and open issues for current development priorities.

---

## **References and Acknowledgments**

This work builds upon extensive research in game AI and deep reinforcement learning. See **[references.md](documentation/references.md)** for comprehensive citations and related work.

Special recognition to the AlphaZero team at DeepMind for the foundational research that made this project possible.

---

## **License**

This project is licensed under the [MIT License](./LICENSE), enabling both academic and commercial use.

---

*Demonstrating that sophisticated AI techniques can be accessible on modest personal budgets.*
