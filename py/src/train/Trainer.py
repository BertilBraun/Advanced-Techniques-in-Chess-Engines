from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Protocol
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm
from torch.amp import GradScaler, autocast

from src.Network import Network
from src.settings import log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import TrainingStats
from src.util.log import log
from src.util.timing import timeit
from src.value import scalar_to_wdl, wdl_to_scalar


class _LossResult(NamedTuple):
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    total_loss: torch.Tensor
    value_output: torch.Tensor


TrainingBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TrainingBatchLoader(Protocol):
    def __iter__(self) -> Iterator[TrainingBatch]: ...

    def __len__(self) -> int: ...


class _LogitForward(nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.logit_forward(state)


def prefetch_training_batches(batches: TrainingBatchLoader) -> Iterator[TrainingBatch]:
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='training-batch') as executor:
        iterator = iter(batches)
        pending_batch = executor.submit(next, iterator)
        while True:
            try:
                batch = pending_batch.result()
            except StopIteration:
                return
            pending_batch = executor.submit(next, iterator)
            yield batch


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingParams,
        training_model: nn.Module | None = None,
        rank: int = 0,
    ) -> None:
        self.model: Network = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.args: TrainingParams = args
        self.training_model = _LogitForward(model) if training_model is None else training_model
        self.rank = rank

    def _calculate_loss_for_batch(self, batch: TrainingBatch) -> _LossResult:
        """Calculate losses for a single batch"""
        state, policy_targets, value_targets = batch

        # Move to device
        state = state.to(device=self.model.device)
        policy_targets = policy_targets.to(device=self.model.device)
        value_targets = value_targets.to(device=self.model.device)

        # Forward pass
        policy_logits, value_logits = self.training_model(state)
        value_probabilities = torch.softmax(value_logits, dim=1)
        value_output = wdl_to_scalar(value_probabilities).unsqueeze(1)

        # Binary cross entropy loss for the policy is definitely not correct, as the policy has multiple classes
        # torch.cross_entropy applies softmax internally, so we don't need to apply it to the output
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        # KL divergence and cross entropy are in principle equivalent, but cross entropy is more numerically stable and slightly faster
        # policy_loss = F.kl_div(F.log_softmax(out_policy, dim=1), policy_targets, reduction='batchmean')
        # MSE loss can work for policy targets, but it is not ideal and does not converge as well as cross entropy
        # policy_loss = F.mse_loss(F.softmax(out_policy, dim=1), policy_targets)

        value_targets = scalar_to_wdl(value_targets)
        value_loss = F.cross_entropy(value_logits, value_targets)

        total_loss = self.args.policy_loss_weight * policy_loss + self.args.value_loss_weight * value_loss

        return _LossResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            value_output=value_output,
        )

    def _train_epoch(self, dataloader: TrainingBatchLoader) -> TrainingStats:
        """Train for one epoch and return statistics"""
        self.model.train()
        self.training_model.train()
        reduction_values = torch.zeros(9, device=self.model.device, dtype=torch.float64)
        scaler = GradScaler(self.model.device.type, enabled=self.model.device.type == 'cuda')

        for batch in tqdm(
            prefetch_training_batches(dataloader),
            total=len(dataloader),
            desc='Train batches',
            disable=self.rank != 0,
        ):
            self.optimizer.zero_grad()
            sample_count = batch[0].shape[0]

            with autocast(self.model.device.type, dtype=torch.bfloat16):
                policy_loss, value_loss, total_loss, value_output = self._calculate_loss_for_batch(batch)

            # Backward pass with scaling
            scaler.scale(total_loss).backward()

            # Gradient clipping with unscaling
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            scaler.step(self.optimizer)
            scaler.update()

            reduction_values[0] += policy_loss.detach().double() * sample_count
            reduction_values[1] += value_loss.detach().double() * sample_count
            reduction_values[2] += total_loss.detach().double() * sample_count
            reduction_values[3] += sample_count
            reduction_values[4] += value_output.detach().double().sum()
            reduction_values[5] += value_output.detach().double().square().sum()
            if self.rank == 0:
                reduction_values[6] += grad_norm.detach().double()
                reduction_values[7] += 1
                reduction_values[8] += 1

        if distributed.is_initialized():
            distributed.all_reduce(reduction_values, op=distributed.ReduceOp.SUM)

        return TrainingStats(
            policy_loss_sum=float(reduction_values[0].item()),
            value_loss_sum=float(reduction_values[1].item()),
            total_loss_sum=float(reduction_values[2].item()),
            sample_count=int(reduction_values[3].item()),
            value_sum=float(reduction_values[4].item()),
            value_square_sum=float(reduction_values[5].item()),
            gradient_norm_sum=float(reduction_values[6].item()),
            gradient_norm_count=int(reduction_values[7].item()),
            num_batches=int(reduction_values[8].item()),
        )

    @timeit
    def train(
        self,
        dataloader: TrainingBatchLoader,
        iteration: int,
    ) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained with cross-entropy losses for policy and WDL value targets.
        """
        # Set learning rate
        base_lr: float = self.args.learning_rate(iteration, self.args.optimizer)
        if self.rank == 0:
            log_scalar('training/learning_rate', base_lr, iteration)
            log(f'Setting learning rate to {base_lr} for iteration {iteration}')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        return self._train_epoch(dataloader)
