from __future__ import annotations

from types import TracebackType

import numpy as np
import torch
import torch.distributed as distributed
from src.games.contracts import GameStateContract
from src.replay.description import ReplayDescription
from src.replay.head_batch import SearchBudgetHeadBatch, SearchBudgetLabelPool
from src.search_budget.configuration import SearchBudgetHeadTrainingConfiguration
from src.training.configuration import TrainingPrecision
from src.training.distributions import ScalarAuxiliaryTrainingDistribution
from src.training.network import Network
from src.training.trainer.contracts import SearchBudgetHeadStatistics
from torch import nn
from torch.nn import functional

# Keeps head-batch sampling independent of the main replay sampler for the same quantum.
_HEAD_BATCH_SEED_SALT = 0x5EA4C8
_DISTRIBUTION_ROWS = 256


class SearchBudgetHeadTrainer:
    """Trains the search-budget head on fully labelled batches, with the shared trunk frozen.

    The labelled pool is a skewed slice of replay, so letting it reach the trunk would bend the whole network
    towards the label sample; freezing the trunk also drops the cost of a head step to a single forward pass.
    """

    def __init__(
        self,
        replay: ReplayDescription,
        state: GameStateContract,
        auxiliary_index: int,
        configuration: SearchBudgetHeadTrainingConfiguration,
        loss_weight: float,
        model: Network,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        precision: TrainingPrecision,
        maximum_gradient_norm: float,
        world_size: int,
        rank: int,
        sampler_seed: int,
        source_optimizer_step: int,
    ) -> None:
        self.configuration = configuration
        self.auxiliary_index = auxiliary_index
        self.loss_weight = loss_weight
        self.model = model
        self.head_module: nn.Module = model.auxiliary_head_modules[auxiliary_index]
        self.head_parameters = tuple(self.head_module.parameters())
        self.optimizer = optimizer
        self.device = device
        self.precision = precision
        self.maximum_gradient_norm = maximum_gradient_norm
        self.world_size = world_size
        self.rank = rank
        self.sampler_seed = sampler_seed
        self.source_optimizer_step = source_optimizer_step
        self.pool = SearchBudgetLabelPool(replay, state, auxiliary_index)
        self.state = state
        self.global_batch_rows = self._agreed_global_batch_rows()
        self.labelled_pool_rows = self.pool.size
        self._loss_sum = 0.0
        self._target_sum = 0.0
        self._target_square_sum = 0.0
        self._prediction_sum = 0.0
        self._prediction_square_sum = 0.0
        self._absolute_error_sum = 0.0
        self._row_count = 0
        self._steps = 0
        self._distribution: ScalarAuxiliaryTrainingDistribution | None = None

    @property
    def local_batch_rows(self) -> int:
        return self.global_batch_rows // self.world_size

    def due_at_step(self, batch_index: int) -> bool:
        return self.global_batch_rows > 0 and batch_index % self.configuration.interval_optimizer_steps == 0

    def train_step(self, batch_index: int, capture_distribution: bool) -> None:
        batch = self._next_batch(batch_index).to_device(self.device, non_blocking=False)
        predictions, absolute_error = self._forward(batch)
        loss = absolute_error.mean()
        gradients = torch.autograd.grad(self.loss_weight * loss, self.head_parameters)
        if self.world_size > 1:
            for gradient in gradients:
                distributed.all_reduce(gradient)
                gradient.div_(self.world_size)
        for parameter, gradient in zip(self.head_parameters, gradients, strict=True):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(self.head_parameters, self.maximum_gradient_norm)
        self.optimizer.step()
        self._accumulate(batch.targets, predictions, absolute_error, loss)
        if capture_distribution:
            self._distribution = _scalar_distribution(batch.targets, predictions, absolute_error)

    @property
    def distribution(self) -> ScalarAuxiliaryTrainingDistribution | None:
        return self._distribution

    def statistics(self) -> SearchBudgetHeadStatistics:
        rows = float(self._row_count)
        steps = float(self._steps)
        target_mean = self._target_sum / rows if rows else 0.0
        prediction_mean = self._prediction_sum / rows if rows else 0.0
        return SearchBudgetHeadStatistics(
            auxiliary_index=self.auxiliary_index,
            labelled_pool_rows=self.labelled_pool_rows,
            global_batch_rows=self.global_batch_rows,
            optimizer_steps=self._steps,
            loss=self._loss_sum / steps if steps else 0.0,
            target_mean=target_mean,
            target_standard_deviation=_standard_deviation(self._target_square_sum, target_mean, rows),
            prediction_mean=prediction_mean,
            prediction_standard_deviation=_standard_deviation(self._prediction_square_sum, prediction_mean, rows),
            absolute_error_mean=self._absolute_error_sum / rows if rows else 0.0,
        )

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> SearchBudgetHeadTrainer:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _agreed_global_batch_rows(self) -> int:
        # Ranks index the pool independently, so the row count has to be agreed before any collective depends on it.
        available = self.pool.size
        if self.world_size > 1:
            shared = torch.tensor([available], dtype=torch.int64, device=self.device)
            distributed.all_reduce(shared, op=distributed.ReduceOp.MIN)
            available = int(shared[0])
        return resolved_global_batch_rows(self.configuration, available, self.world_size)

    def _next_batch(self, batch_index: int) -> SearchBudgetHeadBatch:
        generator = np.random.default_rng(
            np.random.SeedSequence((self.sampler_seed, self.source_optimizer_step, batch_index, _HEAD_BATCH_SEED_SALT))
        )
        global_indices = self.pool.select_logical_indices(generator, self.global_batch_rows)
        global_augmentations = np.asarray(
            generator.integers(0, self.state.augmentation_count, size=self.global_batch_rows),
            dtype=np.int64,
        )
        start = self.rank * self.local_batch_rows
        stop = start + self.local_batch_rows
        return self.pool.batch(global_indices[start:stop], global_augmentations[start:stop])

    def _forward(self, batch: SearchBudgetHeadBatch) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.model.training
        self.model.eval()
        try:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.precision is TrainingPrecision.BFLOAT16,
                ),
            ):
                features = self.model.trunk_features(batch.states)
        finally:
            self.model.train(was_training)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.precision is TrainingPrecision.BFLOAT16,
        ):
            logits = self.head_module(features)
        predictions = torch.sigmoid(logits.float()).squeeze(1)
        return predictions, functional.l1_loss(predictions, batch.targets, reduction='none')

    def _accumulate(
        self,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        absolute_error: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        detached_targets = targets.detach()
        detached_predictions = predictions.detach()
        self._loss_sum += float(loss.detach())
        self._target_sum += float(detached_targets.sum())
        self._target_square_sum += float((detached_targets * detached_targets).sum())
        self._prediction_sum += float(detached_predictions.sum())
        self._prediction_square_sum += float((detached_predictions * detached_predictions).sum())
        self._absolute_error_sum += float(absolute_error.detach().sum())
        self._row_count += int(targets.shape[0])
        self._steps += 1


def resolved_global_batch_rows(
    configuration: SearchBudgetHeadTrainingConfiguration,
    labelled_rows: int,
    world_size: int,
) -> int:
    """A head batch never carries unlabelled filler: a short pool shrinks the batch instead of padding it."""
    if labelled_rows < configuration.minimum_labelled_rows:
        return 0
    return min(configuration.batch_size, labelled_rows) // world_size * world_size


def _standard_deviation(square_sum: float, mean: float, rows: float) -> float:
    if rows <= 0.0:
        return 0.0
    return float(np.sqrt(max(square_sum / rows - mean * mean, 0.0)))


def _scalar_distribution(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    absolute_error: torch.Tensor,
) -> ScalarAuxiliaryTrainingDistribution:
    rows = slice(0, min(int(targets.shape[0]), _DISTRIBUTION_ROWS))
    return ScalarAuxiliaryTrainingDistribution(
        kind='search_budget',
        target=_floats(targets[rows]),
        prediction=_floats(predictions[rows]),
        absolute_error=_floats(absolute_error[rows]),
    )


def _floats(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu())
