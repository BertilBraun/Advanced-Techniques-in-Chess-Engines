# Self-Play Pre-Training System

Self-Play games at the beginning of a training run are only marginally better than noise. This is because the model is not trained yet and therefore predicts bad moves as well as bad evaluations. This mostly fixes itself after a few iterations, but it can be sped up by using grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This is not realy in the spirit of AlphaZero, but it is a good way to bootstrap the training process with less computational resource requirements.

## **Solution:**

My idea to overcome this problem is to use grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This way, we can train the model with good data from the beginning and therefore improve the self-play games. This should lead to a better model and therefore better self-play games. After a few  iterations, the model should be good enough to generate good training data by itself, so that the self improvement loop can start.

- Grandmaster games can be found here: [https://database.nikonoel.fr/](https://database.nikonoel.fr/)
- Lichess evaluations can be found here: [https://database.lichess.org/#evals](https://database.lichess.org/#evals)
- Stockfish can be found here: [https://stockfishchess.org/download/](https://stockfishchess.org/download/)

## **Training Data Generation:**

(We are using Stockfish 8 on the cluster, because it is the only version that compiles there)

With parallelization, we can generate the training data for the model in a reasonable amount of time. We can use the cluster to generate the training data for the model. We generated about 28mil samples from mainly the lichess evaluations over 2h on 32 threads of the cluster.

## **Results:**

TBD
