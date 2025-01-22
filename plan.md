New plan for the Project:

First: 

- [x] Properly document everything in the project, probably in Markdown, possible spread over multiple files.
- [x] Cleanup the root folder, remove the old folders (move into a branch), and create a new folder structure
- [ ] Keep Branches of the architectures tried in python
- [x] Keep a readme why the architecture was tried and what the results were
- [x] Make a list of all remaining tasks
- [ ] Actually time the executions of the different parts of the project in code (decorator?)
- [ ] Then start by moving the current project to a py/ folder and start the new project in a cpp/ folder
- [ ] Move settings into games folders? In src.settings just import the settings from the games?



DONE something About the current implementation is wrong if it doesnt even learn tictactoe
DONE evaluate against a database of "pro-Moves" !!! High prio

DONE continue to save all These logs which are now produced

DONE log everything to tensorboard, also stuff like the Timings, game gen times, generated games, generated samples etc
DONE time more of the entire project



TODO build up a database of pro-Moves for Chess automatically

DONE build in the reuse of the search trees for the next Iteration
DONE document that





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