import torch
import random
import numpy as np
import tensorflow as tf
import torch.nn.functional as F

from tqdm import tqdm

from src.util import batched_iterate
from src.Network import Network
from src.settings import TORCH_DTYPE
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.alpha_zero.SelfPlay import SelfPlayMemory
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

        train_stats = TrainingStats()
        base_lr = self.args.training.learning_rate(iteration)
        tf.summary.scalar('learning_rate', base_lr, step=iteration)

        self.model.train()

        # TODO can we load the samples in training format? - Alternatively at least not convert them from Tensors to SelfPlayMemory

        validation_batch = memory[: self.args.training.batch_size]
        memory = memory[self.args.training.batch_size :]

        def calculate_loss_for_batch(batch: list[SelfPlayMemory]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state = [mem.state for mem in batch]
            policy_targets = [mem.policy_targets for mem in batch]
            value_targets = [[mem.value_target] for mem in batch]

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets),
            )

            state = torch.tensor(state, dtype=TORCH_DTYPE, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=TORCH_DTYPE, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=TORCH_DTYPE, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            return policy_loss, value_loss, loss

        for batchIdx, sample in tqdm(
            enumerate(batched_iterate(memory, self.args.training.batch_size)),
            desc='Training batches',
            total=len(memory) // self.args.training.batch_size,
        ):
            policy_loss, value_loss, loss = calculate_loss_for_batch(sample)

            # Update learning rate before stepping the optimizer
            lr = self.args.training.learning_rate_scheduler(
                (batchIdx * self.args.training.batch_size) / len(memory), base_lr
            )
            tf.summary.scalar(f'learning_rate/iteration_{iteration}', lr, step=batchIdx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        with torch.no_grad():
            validation_stats = TrainingStats()
            validation_policy_loss, validation_value_loss, validation_loss = calculate_loss_for_batch(validation_batch)
            validation_stats.update(validation_policy_loss.item(), validation_value_loss.item(), validation_loss.item())
            log(f'Validation stats: {validation_stats}')

        return train_stats
