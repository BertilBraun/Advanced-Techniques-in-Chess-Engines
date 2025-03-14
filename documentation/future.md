# Future Work

## Model Quantization for Inference

Int8 for inference. In that case the trainer and self play nodes need different models and after training the model needs to be quantized but apparently up to 4x faster inference [docs](https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization).

## Better Network Architecture

[Deep Reinforcement Learning for Crazyhouse](https://ml-research.github.io/papers/czech2019deep.pdf#page=16) introduced RISEv2 and later apparently RISEv3 which have almost twice the inference speed and significantly less parameters while having the same performance.

## Possibly better Value Targets using Path Consistency

Path Consistency ([paper](https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf)) - seems to be more sample efficient by using the information of the mcts search tree for value targets (5 most recent history states, argmax path in mcst search tree, 2x the mse(v-mean(v of paths)) and 1x the mse(f_v-mean(f_v of history states)) i.e. the feature vector before mapping to the value head)

## Better Hyperparameter Optimization using Population Based Training

Population Based Training ([paper](https://arxiv.org/abs/2003.06212)) - seems to be a better hyperparameter optimization technique than bayesian optimization based on their paper. Apparently they only require a single training run instead of the multiple runs for bayesian optimization.

## Faster Initial learning

Initially, the first iterations of training are almost akin to noise, as the model is not yet trained. In addition, initially the model does not need to learn a lot, just the basics to improve its performance. Therefore the first iterations of training can be sped up by using a smaller model, which allows for faster self-play games and faster training and a model that is only able to learn the basics. Once the model has learned the basics, the model can be scaled up to a larger model, which can learn more complex patterns. To transfer the knowledge from the smaller model to the larger model, the latest samples from the smaller model can be used to train the larger model until the loss of the larger model is lower than the loss of the smaller model. After that, the larger model can be used for the next iterations of training.

## Resignation handling

The current resignation handling is very simple and only resigns if the model is certain that it will lose the game. Two possible improvements are:

- Automatic resignation threshold - play out ~10% of games which should have been resigned and verify, that the percentage of games that could have been won is < 5%, otherwise resign earlier.
- Ensure to play out enough endgames, so that the model is trained on a diverse dataset that also includes endgames. The current models struggle with endgames, as they are not played out until the very end.

## Model Training

The training of AZ and similar work does not directly work on iterational datasets but rather train on some sample from the self play games, they also use a way larger batch size than the 128-256 batch size used here and they use a rolling window of the last 500k games. Should be tried out to see if it improves the training speed and the performance of the model?

## Further Games

- [Othello](https://de.wikipedia.org/wiki/Othello_(Spiel))
- [Gobang](https://de.wikipedia.org/wiki/Gobang)
