# MCTS Optimization

## Early Resignation

To speed up the self-play games, the games are not played out until the very end, but rather a resignation threshold is used. If the model is certain that it will lose the game, the model resigns and the game is not played out until the very end. This way, the self-play games are faster and the training samples are generated faster.

## Fast Playouts

Only playout a portion of the moves with the entire search limits. I.e. choose a random subset of ~20% of the boards on which to do a full search. The rest of the boards are only searched for 1/10th of the time. Don't use the fast playouts as training data, but only keep the full playouts for training. This way, the model is trained on full games, subsampled, to not overfit, and still have good policy training targets as the full playouts remain. The fast playouts are only used for Self-Play to progress the games faster. This is a trade-off between speed and quality of the training data. The fast playouts are not used for training, but only for self-play.

## Faster MCTS Search

In my opinion the MCTS tree of the most visited child should be the most explored already. Why discard all of that prior work when starting the search for that child again? Why not retain that information and continue work from there. Issue with that: The noise which initially gets added to the root priors. Idea: Update that later, once the node is selected as the new root. Then ensure that each node gets an appropriate amount of fixes searches, so that the noise exploration has chance to take effect. I.e. child 2 is selcted. Then the next MCTS search starts with child 2 as the new root node and all of the accompanied expansion and search already done (might need to flip some signs of the scores, not sure). Since these moves were not expanded using a noisy policy (for exploration), we have to remedy that, by adding some noise on the priors of the children of child 2 (the new root). Then we should ensure, that each move gets an appropriate amount of fixes searches (f.e. 10) so that the noise has a chance to take effect. Afterwards, the assumption is, that the passed root node and a fully simulated root node are mostly identical, but building them is way cheaper.
