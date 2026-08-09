# Self-Play Pre-Training System

Self-Play games at the beginning of a training run are only marginally better than noise. This is because the model is not trained yet and therefore predicts bad moves as well as bad evaluations. This mostly fixes itself after a few iterations, but it can be sped up by using grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This is not realy in the spirit of AlphaZero, but it is a good way to bootstrap the training process with less computational resource requirements.

## Solution

My idea to overcome this problem is to use grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This way, we can train the model with good data from the beginning and therefore improve the self-play games. This should lead to a better model and therefore better self-play games. After a few  iterations, the model should be good enough to generate good training data by itself, so that the self improvement loop can start.

- Grandmaster games can be found here: [https://database.nikonoel.fr/](https://database.nikonoel.fr/)
- Lichess evaluations can be found here: [https://database.lichess.org/#evals](https://database.lichess.org/#evals)
- Stockfish can be found here: [https://stockfishchess.org/download/](https://stockfishchess.org/download/)

## Training Data Generation

With parallelization, we can generate the training data for the model in a reasonable amount of time. We can use the cluster to generate the training data for the model. We generated about 28mil samples from mainly the lichess evaluations over 2h on 32 threads of the cluster.

## Approach Implemented

We currently have two scripts eployed in the RL pipeline, one to generate a evaluation set of professional games based on the Grandmaster database from [https://database.nikonoel.fr/](https://database.nikonoel.fr/) (see [ChessDatabase.py](../../py/src/games/chess/ChessDatabase.py)) and a script to train on a set of our Database formats (see [DatasetTrainer.py](../../py/src/eval/DatasetTrainer.py)). These can be used in conjunction to generate a training set from professional games and train on them. This is also useful for debugging and testing the training pipeline as well as to setup a baseline for the training process, which the RL pipeline can be compared to.

## Results

Training a 12 layer model with 128 filters per layer (4.7M parameters) on the generated training data from the past 15 months of professional games for approximately 15M training samples took about 2h on a single A100 GPU. The model was able to achieve a policy accuracy of 43.42% @1, 82.51% @5, and 92.10% @10 after 8 iterations. The value loss was 0.72294.

Initially (during the first 3 iterations), the model value head 'collapsed' to always predict 0. Though this seems to have fixed itself after a few iterations. The value head is now predicting a mean of -0.0029 and a standard deviation of 0.3566. This is a good sign, as the value head is now predicting a reasonable value for the game state.

Value count in range (of a 512 batch):

| Value range  | Count |
| ------------ | ----- |
| (-1.0, -0.9) | 1     |
| (-0.9, -0.8) | 5     |
| (-0.8, -0.7) | 5     |
| (-0.7, -0.6) | 12    |
| (-0.6, -0.5) | 16    |
| (-0.5, -0.4) | 27    |
| (-0.4, -0.3) | 32    |
| (-0.3, -0.2) | 35    |
| (-0.2, -0.1) | 39    |
| (-0.1, 0.0)  | 76    |
| (0.0, 0.1)   | 95    |
| (0.1, 0.2)   | 44    |
| (0.2, 0.3)   | 36    |
| (0.3, 0.4)   | 18    |
| (0.4, 0.5)   | 12    |
| (0.5, 0.6)   | 19    |
| (0.6, 0.7)   | 14    |
| (0.7, 0.8)   | 19    |
| (0.8, 0.9)   | 0     |
| (0.9, 1.0)   | 0     |
| (1.0, 1.1)   | 0     |

Compared against Stockfish 15, it plays on equal time settings at Skill Level 3-4.
This seems reasonable given: <https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/experiments/html/train_all_games_over_2000_elo/SGD>
