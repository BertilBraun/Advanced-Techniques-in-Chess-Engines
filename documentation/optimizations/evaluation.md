# Evaluation

## System Usage Logger

We developed a system usage logger, which logs the CPU and RAM usage of each process and the GPU usage of all GPUs. This way, we can monitor the system usage during training and optimize the system usage. The system usage logger logs the system usage to a file and the data can be visualized in a graph. This way, we can monitor the system usage during training and optimize the system usage.

## Statistics Gathering and Monitoring

Keeping statistics about the training process, like the win rate against different opponents, the loss of the model, the training speed, the inference speed, the system usage, resigantion rate, log the played games so that they can be analyzed later, win rate vs stronger baselines (f.e. Stockfish on different levels (4)) etc. This way, the training process can be monitored and optimized.

## Profiling

Using `viztracer`, the code was profiled to find bottlenecks and optimize the code. Especially the MCTS search is a constant bottleneck and was optimized to be faster. The parallelization techniques applicable to the MCTS search were applied but the Python GIL was a constant bottleneck which does not allow for full parallelization. A `Cython` implementation of the MCTS search could be a possible optimization, but was not pursued in this project. A `C++` implementation of the MCTS search stands to be the most promising optimization, and might be pursued in the future (see [Self-Play Problem](inference.md#torch-optimizations)).

## Model Performance Evaluation

To evaluate the performance of the model, the model is played against different opponents every N iterations. The performance of the model is then evaluated based on the win rate against the different opponents half of the games are played as white and half of the games are played as black. Each player has a time limit of 200ms per move or 60 searches per move. We did not use these evaluations to select the best model as AlphaGoZero does, but rather to evaluate the performance of the model. The models are continuously trained and evaluated and the performance of the model is monitored. The opponents are:

### Model vs Random

A random bot is used as a baseline to evaluate the performance of the model. The random bot selects a random move from the legal moves. The model should be able to beat the random bot with a perfect win rate after just a few iterations.

### Model vs HandcraftedBot / Stockfish

A handcrafted bot or Stockfish is used as a baseline to evaluate the performance of the model. The handcrafted bot or Stockfish selects the best move based on a set of rules from a simple minimax search or a more complex search algorithm. The complexity of the handcrafted bot or Stockfish can be adjusted to evaluate the performance of the model.

### Model vs Previous Iteration

The model is played against the previous evaluation iteration to evaluate the performance of the model. Initially, the model should start out by learning a lot each iteration and the win rate should increase with each iteration. After a certain number of iterations, the win rate should stabilize and as each model plays well, the win rate should be around 50%. A continuous decrease in the win rate might indicate that the model is overfitting or that the training data is not diverse enough.

### Model vs Pro-Database

The model is evaluated without search against a database of (near)-optimal moves. The model should achieve a policy accuracy in the high 90s. This is a good indicator of the model's performance.

As a baseline performance reference, you can use the `src.eval.DatasetTrainer` to train a model on the Pro-Database and evaluate it against the Pro-Database. This way, you can see how well the model trained on perfect data performs. This way, you can see wheather the self-play games even come close to the Pro-Database in terms of policy accuracy.
