from typing import NamedTuple
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from src.Network import Network
from src.settings import log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import log
from src.util.timing import timeit


class _LossResult(NamedTuple):
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    total_loss: torch.Tensor
    value_output: torch.Tensor


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingParams,
    ) -> None:
        self.model: Network = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.args: TrainingParams = args

    def _calculate_loss_for_batch(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> _LossResult:
        """Calculate losses for a single batch"""
        state, policy_targets, value_targets = batch

        # Move to device
        state = state.to(device=self.model.device)
        policy_targets = policy_targets.to(device=self.model.device)
        value_targets = value_targets.to(device=self.model.device)

        value_targets = value_targets.unsqueeze(1)

        # Forward pass
        policy_logits, value_logits = self.model.logit_forward(state)
        value_output = torch.tanh(value_logits)

        # Binary cross entropy loss for the policy is definitely not correct, as the policy has multiple classes
        # torch.cross_entropy applies softmax internally, so we don't need to apply it to the output
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        # KL divergence and cross entropy are in principle equivalent, but cross entropy is more numerically stable and slightly faster
        # policy_loss = F.kl_div(F.log_softmax(out_policy, dim=1), policy_targets, reduction='batchmean')
        # MSE loss can work for policy targets, but it is not ideal and does not converge as well as cross entropy
        # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1), policy_targets)

        # NOTE: At |logit| ≈ 4 the derivative of tanh is already < 0.002, so the gradient almost vanishes.
        # Smoothing the value targets to avoid overfitting to the extreme values which no longer give sensible gradients
        # Non-saturated logits: 0.95 corresponds to logit ≈ +2.94, -0.95 to logit ≈ −2.94 – well inside the linear part of tanh. The gradient w.r.t. the logits is therefore still ≈ 0.05–0.1 instead of ≈ 0.001.
        # Therefore clip the value targets to the range [-0.95, 0.95] to avoid saturation effects
        value_targets = torch.clamp(value_targets, min=-0.95, max=0.95)

        # BCE is not suitable for regression tasks and produces spikey outputs at the extremes, so we use MSE instead
        # value_targets = (value_targets + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1] range for binary cross entropy
        # value_loss = F.binary_cross_entropy_with_logits(value_logits, value_targets)
        # value_loss = F.l1_loss(out_value, value_targets)  # l1_loss = mean_absolute_error
        value_loss = F.mse_loss(value_output, value_targets)

        total_loss = self.args.policy_loss_weight * policy_loss + self.args.value_loss_weight * value_loss

        return _LossResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            value_output=value_output,
        )

    def _train_epoch(self, dataloader: DataLoader) -> TrainingStats:
        """Train for one epoch and return statistics"""
        self.model.train()
        stats = TrainingStats(self.model.device)
        scaler = GradScaler()

        for batch in tqdm(dataloader, desc='Train batches'):
            self.optimizer.zero_grad()

            with autocast(self.model.device.type, dtype=torch.bfloat16):
                policy_loss, value_loss, total_loss, value_output = self._calculate_loss_for_batch(batch)

            # Backward pass with scaling
            scaler.scale(total_loss).backward()

            # Gradient clipping with unscaling
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            scaler.step(self.optimizer)
            scaler.update()

            # Collect statistics
            stats.add_batch(policy_loss, value_loss, total_loss, value_output, grad_norm.item())

        return stats

    def _validate_epoch(self, validation_dataloader: DataLoader) -> TrainingStats:
        """Validate for one epoch and return statistics"""
        self.model.eval()
        stats = TrainingStats(self.model.device)

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type=self.model.device.type):
            for batch in tqdm(validation_dataloader, desc='Valid batches'):
                policy_loss, value_loss, total_loss, value_output = self._calculate_loss_for_batch(batch)

                # Collect statistics (no gradient norm for validation)
                stats.add_batch(policy_loss, value_loss, total_loss, value_output)

        return stats

    @timeit
    def train(
        self, dataloader: DataLoader, validation_dataloader: DataLoader, iteration: int
    ) -> tuple[TrainingStats, TrainingStats]:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        # Set learning rate
        base_lr: float = self.args.learning_rate(iteration, self.args.optimizer)
        log_scalar('training/learning_rate', base_lr, iteration)
        log(f'Setting learning rate to {base_lr} for iteration {iteration}')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        # Training phase
        train_stats: TrainingStats = self._train_epoch(dataloader)

        # Validation phase
        validation_stats: TrainingStats = self._validate_epoch(validation_dataloader)

        return train_stats, validation_stats
