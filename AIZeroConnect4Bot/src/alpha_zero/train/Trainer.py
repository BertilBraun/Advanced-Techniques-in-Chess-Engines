import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.utils.data.dataloader
from tqdm import tqdm

from src.alpha_zero.SelfPlayDataset import SelfPlayTrainDataset
from src.Network import Network
from src.settings import log_scalar
from src.alpha_zero.train.TrainingArgs import TrainingParams
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.util.log import log

# DONE AlphaZero simply maintains a single neural network that is updated continually, rather than waiting for an iteration to complete
# DONE do so, save model after each epoch, use smaller num_parallel_games
# DONE save deduplicated dataset for the previous iteration
# DONE simply set num_epochs to 1 and increase the num_iterations, while decreasing how fast the window size grows but increasing the base and max window size
# DONE game and inference nodes? So that 100% GPU is used on inference nodes and as many nodes as needed can supply the self-play nodes
# DONE current MCTS -> BatchedMCTS and create new ClientServerMCTS - based on what protocol?
# DONE make sure to instantly update the inference servers once a new model is available
# DONE infrence server caching
# DONE make sure, that each process logs their CPU and RAM usage and the root logs usage for all GPUs - Display all data and averages in visualization


# DONE something between Deduplication and Training Batches takes 10min?!
# DONE [17.03.24] [INFO] Exception in Training: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

# DONE /memory_19_deduplicated.pt (deflated 96%) Why is the memory able to be compressed so much? What about the memory is so compressible? Yeah.. just a lot of either 0 or 1 values.. Does it make sense to store it as a bit array?
# DONE with the encoded boards, inter process comunication of the states should be way faster and more efficient
# DONE optimize encode and decode with numpy
# DONE load and save via numpy not torch

# NOT_REQUIRED Not always reload each iteration, but only if the memory is not already loaded - only takes a second to load the memory
# DONE single GPU for training, multiple GPUs for self-play?

# DONE use both the final game result as well as the MCTS root value_sum / num_visits as the value target for the NN training. Averaging f.e.

# DONE reduce board size to 6x7

# DONE parallel MCTS search? - searching multiple states at once by blocking nodes: https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf

# FUTURE: Int8 for inference. In that case the trainer and self play nodes need different models and after training the model needs to be quantized but apparently up to 4x faster inference. https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization

# DONE remove or increase number of load balancers if that is a bottleneck
# DONE batched write back from inference server
# DONE read and write with bytes instead of numpy arrays
# DONE profile
# DONE system usage logger

# DONE magic numbers into settings with default values
# DONE default values in settings
# TODO proper documentation
# TODO proper graph representing the different architectures tried
# TODO generate branches for the different architecture approaches tried
# DONE run for Connect4 with new setup for 2h
# DONE run for Checkers
# DONE log time for each self play loop, how long for n games to finish - compare to previous
# DONE fix opt.py
# FUTURE: hyperparameter optimization as in Paper: Accelerating and Improving AlphaZero Using Population Based Training
# FUTURE: start with a small model, then increase the size of the model after some iterations and retrain that model on the old data until the loss is lower than the previous model

# DONEISCH use HandcraftedBotV4 or Stockfish on level 4 as baseline to compare against

# FUTURE: IMO: The MCTS tree of the most visited child should be the most explored already. Why discard all of that prior work when starting the search for that child again? Why not retain that information and continue work from there. Issue with that: The noise which initially gets added to the root priors. Idea: Update that later, once the node is selected as the new root. Then ensure that each node gets an appropriate amount of fixes searches, so that the noise exploration has chance to take effect. I.e. child 2 is selcted. Then the next MCTS search starts with child 2 as the new root node and all of the accompanied expansion and search already done (might need to flip some signs of the scores, not sure). Since these moves were not expanded using a noisy policy (for exploration), we have to remedy that, by adding some noise on the priors of the children of child 2 (the new root). Then we should ensure, that each move gets an appropriate amount of fixes searches (f.e. 10) so that the noise has a chance to take effect. Afterwards, the assumption is, that the passed root node and a fully simulated root node are mostly identical, but building them is way cheaper.

# TODO NOT_REALLY_REQUIRED use mp.Event to signal instead of mp.Pipes?

# NOT_REQUIRED FSDP Data parallel model training
# FUTURE: maybe keep the window based on the number of samples, instead of the number of iterations
# DONE smarter data loading for training, not loading everything in memory at once. How to shuffle that? How to do so with DataLoader and DataParallel?
# Do so by: Assuming, deduplication works in memory. Then we shuffle the deduplicated dataset before writing it to disk. Then we store the data in chunks of ~1GB. Then during training we load only one chunk of each of the iterations datasets and choose the sample based on idx % num_iterations. But then the DataLoader should not shuffle but instead load the samples in order. Yes, that it does, tested.
# DONE save to the dataset how long generating the samples/games took and print these stats while loading the dataset instead of at each save. Then also more frequent dataset saves are possible.

# DONE Caching based on symmetries of the board state, use the smallest key of the symmetries as the key for the cache

# FUTURE: Path Consistency https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf - seems to be more sample efficient by using the information of the mcts search tree for value targets (5 most recent history states, argmax path in mcst search tree, 2x the mse(v-mean(v of paths)) and 1x the mse(f_v-mean(f_v of history states)) i.e. the feature vector before mapping to the value head)


# DONE compare inference speed with and without fusing on both cpu as well as gpu compiled as well as not compiled

# DONE resignation to not play out games until the very end which might require hundreds of moves
# FUTURE: automatic resignation threashold - play out ~10% of games which should have been resigned and verify, that the percentage of games that could have been won is < 5%, otherwise resign earlier
# TODO the endgame currently sucks because no samples will be collected there at all, since one player will always have resigned and the game wont be played out.

# DONE deduplicate same state games in parallel play - if the game is at the same state, sample a different move

# DONE usage during training is also just 40% - let other processes use the GPU as well

# TODO log played games to file, including their moves, scores, resignations, etc. for later analysis

# FUTURE: Othello https://de.wikipedia.org/wiki/Othello_(Spiel)
# FUTURE: Gobang https://de.wikipedia.org/wiki/Gobang


# NOTE Queue based system 2 Inference Servers on 2 GPUS with 40 clients in total
# [15.50.00] [INFO] Generating 1030 samples took 430.68sec

# NOTE old Client inference based system with 12 clients per GPU
# Self Play for 64 games in parallel: 100%|██████████| 1/1 [03:44<00:00, 224.13s/it]
# [12.58.39] [INFO] Collected 2120 self-play memories.
# Approx 220sec for 2120 samples, 110sec for ~1000 samples -> approx 4x faster (But more clients per GPU were used)

# NOTE on pipe based system still has to be recorded

# NOTE new Client inference based system with 12 clients per GPU
# Approx 20sec for 500 samples, 40sec for 1000 samples -> approx 3x faster than old client inference based system and 12x faster than queue based system


# DONE inference server handles with queues and is as light weight as possible
# DONE the load balancer will manage locations of incoming requests, caches, load balancing to the inference servers and proper redistribution of requests to the callers (in the correct order?)
# NOT_NECESSARY remove caching from clients? Do they get the results in the correct order?
# DONE start a new parallel game as soon as a game finishes, not when all games finish? When to check for model/iteration updates?
# DONE optimize MCTS Node again
# NOT_NECESSARY batch on the cache layer already and send complete batches to the inference server

# Alternative approach
# DONE use asyncio to handle many games and search trees in parallel, assembling the requests into a batch and evaluating them locally
# DONE use a local asyncio event to notify once the network inference was done, so that results can be processed and the next iteration can be started
# Drawbacks:
# - less caching possible
# - GPU utilization based on how well the os schedules the processes
# - multiple models per GPU loaded - less vram remaining - lower batch size
# Benefits:
# - simpler architecture - actually more of a drawback if the project should be shown off
# - !!less communication overhead!!


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingParams,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def train(self, dataset: SelfPlayTrainDataset, iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

        train_stats = TrainingStats()
        base_lr = self.args.learning_rate(iteration)
        log_scalar('learning_rate', base_lr, iteration)

        self.model.train()

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            value_targets = value_targets.unsqueeze(1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            return policy_loss, value_loss, loss

        total_policy_loss = torch.tensor(0.0, device=self.model.device)
        total_value_loss = torch.tensor(0.0, device=self.model.device)
        total_loss = torch.tensor(0.0, device=self.model.device)

        for batchIdx, batch in enumerate(tqdm(dataloader, desc='Training batches')):
            if batchIdx == len(dataloader) - 1:
                # Use the last batch as validation batch
                validation_batch = batch
                break

            policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

            # Update learning rate before stepping the optimizer
            batch_percentage = batchIdx / len(dataloader)
            lr = self.args.learning_rate_scheduler(batch_percentage, base_lr)
            log_scalar(f'learning_rate/iteration_{iteration}', lr, batchIdx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss

        train_stats.update(total_policy_loss.item(), total_value_loss.item(), total_loss.item(), len(dataloader))

        with torch.no_grad():
            validation_stats = TrainingStats()
            val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)

            log_scalar('val_policy_loss', val_policy_loss.item(), iteration)
            log_scalar('val_value_loss', val_value_loss.item(), iteration)
            log_scalar('val_loss', val_loss.item(), iteration)
            validation_stats.update(val_policy_loss.item(), val_value_loss.item(), val_loss.item())

            log(f'Validation stats: {validation_stats}')

        return train_stats
