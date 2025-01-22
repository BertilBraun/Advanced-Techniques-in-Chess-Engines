# MCTS Optimization

## Early Resignation

To speed up the self-play games, the games are not played out until the very end, but rather a resignation threshold is used. If the model is certain that it will lose the game, the model resigns and the game is not played out until the very end. This way, the self-play games are faster and the training samples are generated faster.

## Faster MCTS Search

In my opinion the MCTS tree of the most visited child should be the most explored already. Why discard all of that prior work when starting the search for that child again? Why not retain that information and continue work from there. Issue with that: The noise which initially gets added to the root priors. Idea: Update that later, once the node is selected as the new root. Then ensure that each node gets an appropriate amount of fixes searches, so that the noise exploration has chance to take effect. I.e. child 2 is selcted. Then the next MCTS search starts with child 2 as the new root node and all of the accompanied expansion and search already done (might need to flip some signs of the scores, not sure). Since these moves were not expanded using a noisy policy (for exploration), we have to remedy that, by adding some noise on the priors of the children of child 2 (the new root). Then we should ensure, that each move gets an appropriate amount of fixes searches (f.e. 10) so that the noise has a chance to take effect. Afterwards, the assumption is, that the passed root node and a fully simulated root node are mostly identical, but building them is way cheaper.
