import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.Network import Network
from src.settings import TORCH_DTYPE, log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import log
from src.util.timing import timeit


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

    @timeit
    def train(self, dataloader: DataLoader, iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        train_stats = TrainingStats()
        base_lr = self.args.learning_rate(iteration)
        log_scalar('learning_rate', base_lr, iteration)

        self.model.train()

        out_value_mean = torch.tensor(0.0, device=self.model.device)
        out_value_std = torch.tensor(0.0, device=self.model.device)

        def calculate_loss_for_batch(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            state, policy_targets, value_targets = batch

            state = state.to(device=self.model.device, dtype=TORCH_DTYPE, non_blocking=True)
            policy_targets = policy_targets.to(device=self.model.device, dtype=TORCH_DTYPE, non_blocking=True)
            value_targets = value_targets.to(device=self.model.device, dtype=TORCH_DTYPE, non_blocking=True)

            value_targets = value_targets.unsqueeze(1)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            nonlocal out_value_mean, out_value_std
            out_value_mean += out_value.mean()
            out_value_std += out_value.std()

            # Apparently just as in AZ Paper, give more weight to the policy loss
            # loss = torch.lerp(value_loss, policy_loss, 0.66)
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
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss

        train_stats.update(
            total_policy_loss.item(),
            total_value_loss.item(),
            total_loss.item(),
            out_value_mean.item(),
            out_value_std.item(),
            len(dataloader),
        )

        with torch.no_grad():
            validation_stats = TrainingStats()
            val_policy_loss, val_value_loss, val_loss = calculate_loss_for_batch(validation_batch)

            log_scalar('validation/policy_loss', val_policy_loss.item(), iteration)
            log_scalar('validation/value_loss', val_value_loss.item(), iteration)
            log_scalar('validation/loss', val_loss.item(), iteration)
            validation_stats.update(val_policy_loss.item(), val_value_loss.item(), val_loss.item(), 0, 0)

            log(f'Validation stats: {validation_stats}')

        return train_stats
