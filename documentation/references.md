# Research references

The maintained, publication-oriented source list is the
[technical-report bibliography and citation plan](report/bibliography.md). It prioritizes primary research,
versioned project documentation, and repository evidence. The list below preserves the project's original reading
trail while putting the authoritative sources used by the report first.

## Authoritative sources used by the report

### AlphaZero and reproducible implementations

- Silver, D. et al. (2017). [*Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
  Algorithm*](https://arxiv.org/abs/1712.01815).
- Silver, D. et al. (2017). [*Mastering the Game of Go without Human
  Knowledge*](https://doi.org/10.1038/nature24270).
- Tian, Y. et al. (2019). [*ELF OpenGo: An Analysis and Open Reimplementation of
  AlphaZero*](https://arxiv.org/abs/1902.04522).
- Lee, B., Jackson, A., Madams, T., Troisi, S., and Jones, D. (2019). [*Minigo: A Case Study in Reproducing
  Reinforcement Learning Research*](https://openreview.net/pdf?id=H1eerhIpLV).

### KataGo

- Wu, D. J. (2019). [*Accelerating Self-Play Learning in Go*](https://arxiv.org/abs/1902.10565). This is the
  primary source for playout-cap randomization, forced playouts and policy-target pruning, auxiliary targets,
  global-pooling architecture, and the reported efficiency ablations.
- KataGo 1.17.1: [repository](https://github.com/lightvector/KataGo/tree/v1.17.1),
  [additional methods](https://github.com/lightvector/KataGo/blob/v1.17.1/docs/KataGoMethods.md),
  [self-play training](https://github.com/lightvector/KataGo/blob/v1.17.1/SelfplayTraining.md), and
  [training history](https://github.com/lightvector/KataGo/blob/v1.17.1/TrainingHistory.md). These versioned pages
  support implementation and historical claims that are not in the 2019 paper, including policy-surprise weighting.

KataGo's results motivate experiments; they do not prove that a technique helped this chess implementation. Local
claims must cite the corresponding repository benchmark or be labeled retained-without-isolated-ablation.

### Search, replay, and difficult-state work

- Lan, L.-C. et al. (2021). [*Learning to Stop: Dynamic Simulation Monte-Carlo Tree
  Search*](https://arxiv.org/abs/2012.07910).
- Czech, J., Korus, P., and Kersting, K. (2020). [*Monte-Carlo Graph Search for
  AlphaZero*](https://arxiv.org/abs/2012.11045).
- Danihelka, I., Guez, A., Schrittwieser, J., and Silver, D. (2022).
  [*Policy Improvement by Planning with Gumbel*](https://openreview.net/pdf?id=bERaNdoegnO).
- Trudeau, A. and Bowling, M. (2023). [*Targeted Search Control in AlphaZero for Effective Policy
  Improvement*](https://arxiv.org/abs/2302.12359). This is the Go-Exploit restart-state source.
- Schrittwieser, J. et al. (2021). [*Online and Offline Reinforcement Learning by Planning with a Learned
  Model*](https://arxiv.org/abs/2104.06294). This is the primary Reanalyse/MuZero Unplugged reference; reanalysis
  is not part of the final system in this repository.
- Fedus, W. et al. (2020). [*Revisiting Fundamentals of Experience
  Replay*](https://arxiv.org/abs/2007.06700).
- D'Oro, P., Schwarzer, M., Nikishin, E., Bacon, P.-L., Bellemare, M. G., and Courville, A. (2023).
  [*Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio
  Barrier*](https://openreview.net/pdf?id=OpC-9aBBVJe). This is evidence from Atari and control domains, not a
  result about AlphaZero chess.
- Jones, A. L. (2021). [*Scaling Scaling Laws with Board Games*](https://arxiv.org/abs/2104.03113).

## Historical exploratory reading list

The remaining sources are retained for provenance. They include tutorials, practitioner write-ups, and example
repositories. They may be useful for orientation, but should not support technical-report claims when a primary or
versioned authoritative source exists.

### Papers and articles

#### Core papers

* **[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)](https://arxiv.org/pdf/1712.01815)**
  DeepMind’s seminal AlphaZero paper introducing a general reinforcement learning algorithm for board games.

* **[Minigo: A Case Study in Reproducing Reinforcement Learning Research](https://openreview.net/pdf?id=H1eerhIpLV)**
  A practical and reproducible approach to AlphaGo-like systems using TensorFlow.

#### Blog series: *Lessons from AlphaZero* (Oracle Devs)

* **[Connect Four](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68)**
* **[Parameter Tweaking (Part 3)](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)**
* **[Improving the Training Target (Part 4)](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)**
* **[Performance Optimization (Part 5)](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)**
* **[Hyperparameter Tuning (Part 6)](https://medium.com/oracledevs/lessons-from-alphazero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)**

#### Additional resources

* **[AlphaZero Chess: How It Works, What Sets It Apart, and What It Can Tell Us](https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867)**
  An accessible breakdown of AlphaZero’s innovations in chess.

* **[AlphaZero Explained](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)**
  Conceptual overview and key architectural details.

* **[AlphaGo Zero Cheat Sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)**
  A visual summary of the AlphaGo Zero architecture.

* **[AlphaZero from Scratch (YouTube)](https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org)**
  A complete video walkthrough on implementing AlphaZero using Python and PyTorch.

### Repositories

* **[AlphaZero General (by Surag Nair)](https://github.com/suragnair/alpha-zero-general)**
  A flexible, game-independent AlphaZero implementation in Python.

* **[Michael Nny’s AlphaZero](https://github.com/michaelnny/alpha_zero)**
  AlphaZero implementation for Gomoku and Go.

* **[CrazyAra](https://github.com/QueensGambit/CrazyAra)**
  Deep learning model for the Crazyhouse variant of chess.

* **[Minigo (by TensorFlow)](https://github.com/tensorflow/minigo)**
  A scalable, reproducible version of AlphaGo Zero using TensorFlow.

### Parameters and configuration

* **[Gomoku Training Parameters](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/training_gomoku.py)**
* **[Go Training Parameters](https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/training_go.py)**

### Experiments and results

* **[CrazyAra Experiment Summary (CSV)](https://github.com/QueensGambit/CrazyAra/blob/master/DeepCrazyhouse/src/experiments/experiments_summary.csv)**
* **[Minigo Results (TensorFlow)](https://github.com/tensorflow/minigo/blob/6d89c202cdceaf449aefc3149ab2110d44f1a6a4/RESULTS.md)**
* **[Minigo Paper – OpenReview](https://openreview.net/pdf?id=H1eerhIpLV)**

### People and profiles

* **[Johannes  Czech – ML Researcher](https://ml-research.github.io/people/jczech/index.html)**
* **[Johannes  Czech – DeepCrazyhouse Paper (PDF)](https://ml-research.github.io/papers/czech2019deep.pdf)**
