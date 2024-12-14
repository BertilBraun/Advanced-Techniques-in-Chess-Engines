import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from AIZeroConnect4Bot.src.util import batched_iterate
from AIZeroConnect4Bot.src.Network import Network
from AIZeroConnect4Bot.src.settings import TORCH_DTYPE
from AIZeroConnect4Bot.src.train.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.train.TrainingStats import TrainingStats
from AIZeroConnect4Bot.src.self_play.SelfPlay import SelfPlayMemory


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingArgs,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def train(self, memory: list[SelfPlayMemory], iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        random.shuffle(memory)

        train_stats = TrainingStats()
        base_lr = self.args.learning_rate(iteration)

        self.model.train()

        for batchIdx, sample in tqdm(
            enumerate(batched_iterate(memory, self.args.batch_size)),
            desc='Training batches',
            total=len(memory) // self.args.batch_size,
        ):
            state = [mem.state for mem in sample]
            policy_targets = [mem.policy_targets for mem in sample]
            value_targets = [mem.value_targets for mem in sample]

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=TORCH_DTYPE, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=TORCH_DTYPE, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=TORCH_DTYPE, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Update learning rate before stepping the optimizer
            lr = self.args.learning_rate_scheduler(batchIdx / len(memory), base_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        return train_stats
