import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.Network import Network
from src.settings import TB_SUMMARY, TORCH_DTYPE
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
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

    def train(self, dataset: SelfPlayDataset, iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset, [len(dataset) - self.args.training.batch_size, self.args.training.batch_size]
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.training.batch_size,
            shuffle=True,
            drop_last=False,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.args.training.batch_size,
            shuffle=True,
            drop_last=False,
        )

        train_stats = TrainingStats()
        base_lr = self.args.training.learning_rate(iteration)
        TB_SUMMARY.add_scalar('learning_rate', base_lr, iteration)

        self.model.train()

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            state = state.to(device=self.model.device, dtype=TORCH_DTYPE)
            policy_targets = policy_targets.to(device=self.model.device, dtype=TORCH_DTYPE)
            value_targets = value_targets.to(device=self.model.device, dtype=TORCH_DTYPE).unsqueeze(1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            return policy_loss, value_loss, loss

        for batchIdx, batch in tqdm(enumerate(train_dataloader), desc='Training batches', total=len(train_dataloader)):
            policy_loss, value_loss, loss = calculate_loss_for_batch(batch)

            # Update learning rate before stepping the optimizer
            batch_percentage = batchIdx / len(train_dataloader)
            lr = self.args.training.learning_rate_scheduler(batch_percentage, base_lr)
            TB_SUMMARY.add_scalar(f'learning_rate/iteration_{iteration}', lr, batchIdx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        with torch.no_grad():
            validation_stats = TrainingStats()
            for validation_batch in validation_dataloader:
                val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)
                validation_stats.update(val_policy_loss.item(), val_value_loss.item(), val_loss.item())
            log(f'Validation stats: {validation_stats}')

        return train_stats
