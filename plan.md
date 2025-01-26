New plan for the Project:

First: 

- [x] Properly document everything in the project, probably in Markdown, possible spread over multiple files.
- [x] Cleanup the root folder, remove the old folders (move into a branch), and create a new folder structure
- [ ] Keep Branches of the architectures tried in python
- [x] Keep a readme why the architecture was tried and what the results were
- [x] Make a list of all remaining tasks
- [x] Actually time the executions of the different parts of the project in code (decorator?)
- [ ] Then start by moving the current project to a py/ folder and start the new project in a cpp/ folder
- [ ] Move settings into games folders? In src.settings just import the settings from the games?



DONE something About the current implementation is wrong if it doesnt even learn tictactoe
DONE evaluate against a database of "pro-Moves" !!! High prio

DONE continue to save all These logs which are now produced

DONE log everything to tensorboard, also stuff like the Timings, game gen times, generated games, generated samples etc
DONE time more of the entire project
DONE build in the reuse of the search trees for the next Iteration
DONE document that




DONE build up a database of pro-Moves for Chess automatically
DONE Train on that for a baseline Policy and value accuracy

[18.53.33] [INFO] Validation stats: Policy Loss: 1.8125, Value Loss: 0.6758, Total Loss: 2.4844
Epoch 1/1 done: Policy Loss: 1.8466, Value Loss: 0.7082, Total Loss: 2.5548
Evaluation results at iteration 8:
    Policy accuracy @1: 43.42%
    Policy accuracy @5: 82.51%
    Policy accuracy @10: 92.10%
    Avg value loss: 0.7229457942768931
Training with lr: 0.02151582968791256
Training batches: 100%|█████████▉| 29570/29571 [12:51<00:00, 38.31it/s]
Value mean: -0.002927398681640625
Value std: 0.3470703125

Which seems reasonable given: https://github.com/QueensGambit/CrazyAra/tree/master/DeepCrazyhouse/src/experiments/html/train_all_games_over_2000_elo/SGD


DONE faster board hashes - Zobrist Hashing

DONE completely remove asyncio from the project
DONE document the asyncio test

Baseline performance:
Two GPUs were exclusively used for selfplay game generation and one GPU was used
both for game generation and updating the neural network as soon as sufficiently many
samples have been acquired. On a single GPU about 45 games were generated per minute
which corresponded to 1,000 training samples per minute. Each training sample was
produced by an average of 800 MCTS rollouts with a batch-size of 8.

Not all of the game moves were exported as training samples and exported as further
described in Section 5.2.1. A new neural network was generated every 819,200 (= 640 ·
128 · 10) newly generated samples. After the 10th model update, the number of required
samples was increased to 1,228,800 (= 640 · 128 · 15) samples. 81,920 (= 640 · 128) of
these samples were used for validation and the rest for training. Furthermore, 409,600
(= 640 · 128 · 5) samples were added and randomly chosen from 5 % of the most recent
replay memory data. The training proceeded for one epoch and afterwards, all samples,
except validation samples, were added to the replay memory.


DONE fix the OOM issue

TODO Think about, training with the pretrained model or from scratch?
    Value seems stuck on my machine right now
    Another bug?


TODO I think too many positions are duplicated, i.e. searched again, that is why:
- the Cache hit rate is so high
- the model does not learn much (because it only sees a very limited set of positions)
- the gpu usage is low, because no more inference is performed, only hash calculations and Cache lookups


TODO C++ how to handle dependencies?
- libtorch
- xtensor
- json.hpp


Make the project really scalable and in C++

- Translate the Games to C++
- New architecture for the project
  - Per GPU:
    - Several threads for the self players - auto scaling?
      - Each thread will have its own MCTS
      - Each finished game will be written to a file - what format?
    - A caching thread, which will recieve the requests from the MCTS in memory (hopefully fast), look for a cache hit, otherwise perform inference and notify the MCTS
      - Create batches of games to send to the inference thread?
    - A thread for the inference which only performs inference based on the in memory queue from the caching thread
      - Write the entire batch results back to a queue as one result for the caching thread to redistribute
  - One thread for the trainer
  - One thread for the dataloader for the trainer, which will read the files provide the data to the trainer
  - A thread or some notification system to notify the Inference thread to update the model and the caching thread to update the cache and the self players to write the games to the file

- Save in a custom format (binary) by writing out the shape, then the values in order in binary format
  - Save a batch of games -> Write: [batch_size, *game_shape] followed by the games
  - That way both c++ and python can read the files