# Self-Play Pre-Training System

## **Current Model Problems:**

We currently have a problem, that the self-play games at the beginning are not very good. This is because the model is not trained yet and therefore predicts bad moves as well as bad evaluations. This means, that many of the expanded nodes in the MCTS are evaluated by the model instead of the endgame score. This means, that we are training the model with random data, which is not very useful. AlphaZero solves this problem by simply searching more iterations per move, which more often leads to the endgame score being used. However, this is not viable for us, because we are using way less computational resources than AlphaZero.

## **Solution:**

My idea to overcome this problem is to use grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This way, we can train the model with good data from the beginning and therefore improve the self-play games. This should lead to a better model and therefore better self-play games. After a few  iterations, the model should be good enough to generate good training data by itself, so that the self improvement loop can start.

- Grandmaster games can be found here: [https://database.nikonoel.fr/](https://database.nikonoel.fr/)
- Lichess evaluations can be found here: [https://database.lichess.org/#evals](https://database.lichess.org/#evals)
- Stockfish can be found here: [https://stockfishchess.org/download/](https://stockfishchess.org/download/)

## **Training Data Generation:**

(We are using Stockfish 8 on the cluster, because it is the only version that compiles there)

With parallelization, we can generate the training data for the model in a reasonable amount of time. We can use the cluster to generate the training data for the model. We generated about 28mil samples from mainly the lichess evaluations over 2h on 32 threads of the cluster.

To generate the training data, we use the following steps:

```bash
cd train
./data_generator.sh
or
sbatch data_generator.sh
```

The hope is, that the teacher like initialization for the model, will also counteract the problem of the collapsed value head as the model is initially trained on good data with diverse evaluations.

## **Results:**

TBD
