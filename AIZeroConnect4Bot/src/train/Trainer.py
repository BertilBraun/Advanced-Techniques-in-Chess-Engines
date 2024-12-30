import math
import tensorflow as tf
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from src.util import batched_iterate
from src.Network import VALUE_OUTPUT_HEADS, Network
from src.settings import TORCH_DTYPE
from src.train.TrainingArgs import TrainingArgs
from src.train.TrainingStats import TrainingStats
from src.self_play.SelfPlay import SelfPlayMemory
from src.util.TrainingDashboard import TrainingDashboard
from src.util.log import log

# TODO AlphaZero simply maintains a single neural network that is updated continually, rather than waiting for an iteration to complete


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

        train_stats = TrainingStats(self.args.batch_size)
        base_lr = self.args.learning_rate(iteration)
        tf.summary.scalar('learning_rate', base_lr, step=iteration)

        self.model.train()

        # TODO can we load the samples in training format? - Alternatively at least not convert them from Tensors to SelfPlayMemory
        # TODO why does it not learn anything?!
        # - Learning rate is too high? too low?
        # - Samples correct?
        # - MCTS correct? - doesn't seem to be the case with tictactoe

        validation_batch = memory[: self.args.batch_size]
        memory = memory[self.args.batch_size :]

        def calculate_loss_for_batch(batch: list[SelfPlayMemory]):
            state = [mem.state for mem in batch]
            policy_targets = [mem.policy_targets for mem in batch]
            value_targets = [[math.tanh(mem.value_target)] * VALUE_OUTPUT_HEADS for mem in batch]

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets),
            )

            state = torch.tensor(state, dtype=TORCH_DTYPE, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=TORCH_DTYPE, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=TORCH_DTYPE, device=self.model.device)

            out_policy, out_value = self.model(state)
            # out_value is of shape (batch_size, 32), we want to run MSE on each value separately and add the sum to the total loss

            # TODO? policy_loss = F.binary_cross_entropy_with_logits(out_policy, policy_targets)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            # Are mutliple heads really sensible?
            # TODO mult by 10 to make it more impactful??
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # example for value loss
            # targets = torch.tensor([1, 2, 3, 4, 5], dtype=TORCH_DTYPE)
            # outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in range(5)], dtype=TORCH_DTYPE)
            # print(F.mse_loss(outputs, targets, reduction='none'))

            return policy_loss, value_loss, loss

        for batchIdx, sample in tqdm(
            enumerate(batched_iterate(memory, self.args.batch_size)),
            desc='Training batches',
            total=len(memory) // self.args.batch_size,
        ):
            policy_loss, value_loss, loss = calculate_loss_for_batch(sample)

            # Update learning rate before stepping the optimizer
            lr = self.args.learning_rate_scheduler((batchIdx * self.args.batch_size) / len(memory), base_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        with torch.no_grad():
            validation_stats = TrainingStats(self.args.batch_size)
            validation_policy_loss, validation_value_loss, validation_loss = calculate_loss_for_batch(validation_batch)
            validation_stats.update(validation_policy_loss.item(), validation_value_loss.item(), validation_loss.item())
            log(f'Validation stats: {validation_stats}')

        return train_stats
