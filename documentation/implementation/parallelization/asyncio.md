# The Story about Asyncio

To achive the highest inference throughput possible, a large batch size is required. To increase the batch size, multiple approaches can be taken:

- **Parallel Games**: Multiple games can be played in parallel and the inference requests can be batched together. This allows for a larger batch size and therefore a higher throughput. The problem is, that 16 games in 10 minutes is not always better than 1 game in 1 minute, since the training must await the generation of the games.
- **Parallel MCTS**: The MCTS search can be parallelized, so that multiple MCTS searches can be performed in parallel. The problem hereby is, that the MCTS search deterministically selects the node with the highest UCB1 value, which means that the same node will be selected by multiple MCTS searches. To counteract this, we can add a virtual loss to the selected node, so that the other MCTS searches will likely select a different node, unless the UCB1 value of the selected node is significantly higher than the UCB1 value of the other nodes. This approach increases the exploration of the MCTS search and therefore the quality of the MCTS search. A balance must therefore be found, so that the MCTS search is not too explorative while allowing for large batch sizes.

We implemented these concepts using `asyncio` for simplicity, to allow for a clear control flow. During debugging of a large training run, we noticed, that the time spent in our own code diminished to only about 10% of actual runtime (including torch inference calls). This was a surprise and we decided to investigate further.

Using [asyncio_test.py](asyncio_test.py) we found that the overhead of asyncio is about 10x the time of the actual function call.

```text
Iteration - Synchronous Recursive Work
total time spent: 0.59355, total time: 0.59355
total traced time: 100.00%
================================================================================
Iteration - Asynchronous Recursive Work
total time spent: 0.36180, total time: 10.62792
total traced time: 3.40%
```

While the code spend in computation is comparable to the synchronous version, the overhead of asyncio appears to be 96.6% of the total time. This is such a large overhead, that it is not feasible to use asyncio for our purposes. We therefore decided to remove asyncio from the project and to use a different approach to parallelize the MCTS search.
