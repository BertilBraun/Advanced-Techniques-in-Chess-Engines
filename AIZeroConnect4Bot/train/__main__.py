from torch.optim import Adam

from AIZeroConnect4Bot.src.settings import TRAINING_ARGS
from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.Network import Network

"""
Main Documentation used for the project:
- https://arxiv.org/pdf/1712.01815
- https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68
- https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
- https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628
- https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e
- https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a
- https://www.youtube.com/watch?v=wuSQpLinRB4
"""


"""

Multi node rauspflücken und Selfplay und trainer smart mit kleineren args wieder callen


---


Noch offene Profiling bottlenecks?


---


Wie viel von der Performance ist search und wie viel Training?
Benchmarking with very little amount of search terms (like 10 or 20) to see how well just the intuition from the neural net works.


---

Position deduplication
When training the neural net, positions are selected at random amongst the most recent games. Because Connect Four has very few reasonable opening lines, we found this led to a significant overrepresentation of early board positions. This caused the network to focus too much on them.

We mitigated this problem by deduplicating positions before randomly selecting them for training. In the process, we averaged the priors and result values. This seemed to work well.

---

Perhaps the largest direct benefit of this, is that it allowed us to cache position evaluations.

---

Possibly int8 for optimized inference speed as well as using TensorRT if applicable.


---

1 Cycle learning rate policy.
Ramp up from lr/10 to lr over 50% of the batches, then ramp down to lr/10 over the remaining 50% of the batches.
Do this for each epoch separately.


---


We initially implemented the model and its head networks as described in the paper. Based on findings reported by Leela Chess, we increased the number of filters in our head networks to 32, which sped up training significantly.


---


To counteract this, we implemented a slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35.


---


Write out all that the project contians. 




"""

if __name__ == '__main__':
    model = Network()
    optimizer = Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

    print('Starting training')
    print('Training on:', model.device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    print('Training args:', TRAINING_ARGS)
    print('Learning rate:', optimizer.param_groups[0]['lr'])

    AlphaZero(model, optimizer, TRAINING_ARGS).learn()
