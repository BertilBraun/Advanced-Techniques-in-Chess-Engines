# Future Work

## Faster MCTS Search

In my opinion the MCTS tree of the most visited child should be the most explored already. Why discard all of that prior work when starting the search for that child again? Why not retain that information and continue work from there. Issue with that: The noise which initially gets added to the root priors. Idea: Update that later, once the node is selected as the new root. Then ensure that each node gets an appropriate amount of fixes searches, so that the noise exploration has chance to take effect. I.e. child 2 is selcted. Then the next MCTS search starts with child 2 as the new root node and all of the accompanied expansion and search already done (might need to flip some signs of the scores, not sure). Since these moves were not expanded using a noisy policy (for exploration), we have to remedy that, by adding some noise on the priors of the children of child 2 (the new root). Then we should ensure, that each move gets an appropriate amount of fixes searches (f.e. 10) so that the noise has a chance to take effect. Afterwards, the assumption is, that the passed root node and a fully simulated root node are mostly identical, but building them is way cheaper.

## Model Quantization for Inference

Int8 for inference. In that case the trainer and self play nodes need different models and after training the model needs to be quantized but apparently up to 4x faster inference [docs](https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization).

## Possibly better Value Targets using Path Consistency

Path Consistency ([paper](https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf)) - seems to be more sample efficient by using the information of the mcts search tree for value targets (5 most recent history states, argmax path in mcst search tree, 2x the mse(v-mean(v of paths)) and 1x the mse(f_v-mean(f_v of history states)) i.e. the feature vector before mapping to the value head)

## Better Hyperparameter Optimization using Population Based Training

Population Based Training ([paper](https://arxiv.org/abs/2003.06212)) - seems to be a better hyperparameter optimization technique than bayesian optimization based on their paper. Apparently they only require a single training run instead of the multiple runs for bayesian optimization.

## Statistics Gathering and Monitoring

Keep more statistics about the training process, like the win rate against different opponents, the loss of the model, the training speed, the inference speed, the system usage, resigantion rate, log the played games so that they can be analyzed later, win rate vs stronger baselines (f.e. Stockfish on different levels (4)) etc. This way, the training process can be monitored and optimized.

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
