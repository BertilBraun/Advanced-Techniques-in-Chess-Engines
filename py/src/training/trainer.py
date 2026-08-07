from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast

from src.neural_network import Network
from src.training.batch import TrainingBatch
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.util.tensorboard import log_scalar
from src.training.configuration import TrainingParams
from src.training.statistics import (
    EXPECTED_SCORE_CALIBRATION_BINS,
    MATERIAL_VALUE_BIN_LABELS,
    MATERIAL_VALUE_BIN_UPPER_BOUNDS,
    PLY_VALUE_BIN_LABELS,
    PLY_VALUE_BIN_UPPER_BOUNDS,
    TrainingStats,
    ValueMetrics,
)
from src.util.log import log
from src.util.timing import timeit
from src.value import wdl_to_scalar


VALUE_METRIC_WIDTH = 20 + EXPECTED_SCORE_CALIBRATION_BINS * 3
BASE_REDUCTION_WIDTH = 8
SLICED_VALUE_METRIC_BATCH_INTERVAL = 10


@dataclass(frozen=True)
class LossResult:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    total_loss: torch.Tensor
    value_logits: torch.Tensor
    target_expected_scores: torch.Tensor
    value_loss_contributions: torch.Tensor
    final_outcomes: torch.Tensor
    mcts_root_values: torch.Tensor
    outcome_target_eligible: torch.Tensor
    material_result_scores: torch.Tensor
    material_target_eligible: torch.Tensor


@dataclass(frozen=True)
class _DetachedMetricBatch:
    value_logits: torch.Tensor
    target_expected_scores: torch.Tensor
    value_loss_contributions: torch.Tensor
    outcome_target_eligible: torch.Tensor
    material_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor
    final_outcomes: torch.Tensor
    mcts_root_values: torch.Tensor
    material_result_scores: torch.Tensor
    plies: torch.Tensor
    current_player_piece_counts: torch.Tensor
    opponent_piece_counts: torch.Tensor


@dataclass(frozen=True)
class _ValueMetricTensors:
    value_probabilities: torch.Tensor
    expected_scores: torch.Tensor
    target_expected_scores: torch.Tensor
    value_loss_contributions: torch.Tensor
    outcome_losses: torch.Tensor
    mcts_huber_losses: torch.Tensor
    material_huber_losses: torch.Tensor
    outcome_target_eligible: torch.Tensor
    material_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor
    final_outcomes: torch.Tensor
    material_result_scores: torch.Tensor
    plies: torch.Tensor
    current_player_piece_counts: torch.Tensor
    opponent_piece_counts: torch.Tensor


@dataclass(frozen=True)
class _ValueMetricInputs:
    outcome_target_eligible: torch.Tensor
    mcts_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor
    final_outcomes: torch.Tensor
    predicted_classes: torch.Tensor
    target_expected_scores: torch.Tensor
    calibration_bin_indices: torch.Tensor
    brier_scores: torch.Tensor
    expected_score_squared_errors: torch.Tensor
    expected_score_absolute_errors: torch.Tensor


class TrainingBatchLoader(Protocol):
    def __iter__(self) -> Iterator[TrainingBatch]: ...

    def __len__(self) -> int: ...


@dataclass
class DeviceTransferTracker:
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]

    def elapsed_seconds(self) -> float:
        elapsed_milliseconds = 0.0
        for start, end in self.event_pairs:
            end.synchronize()
            elapsed_milliseconds += start.elapsed_time(end)
        return elapsed_milliseconds / 1_000.0


class _LogitForward(nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.logit_forward(state)


class TrainingObjective(ABC):
    @property
    @abstractmethod
    def policy_loss_weight(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def value_loss_weight(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def value_target_weight(self, optimizer_step: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_loss(
        self,
        training_model: nn.Module,
        batch: TrainingBatch,
        device: torch.device,
    ) -> LossResult:
        raise NotImplementedError


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


def prefetch_device_training_batches(
    batches: TrainingBatchLoader,
    device: torch.device,
    transfer_tracker: DeviceTransferTracker,
) -> Iterator[TrainingBatch]:
    cpu_batches = iter(prefetch_training_batches(batches))
    if device.type != 'cuda':
        yield from cpu_batches
        return
    transfer_stream = torch.cuda.Stream(device=device)
    try:
        first_cpu_batch = next(cpu_batches)
    except StopIteration:
        return
    with torch.cuda.stream(transfer_stream):
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        transfer_start.record(transfer_stream)
        current_device_batch = first_cpu_batch.to_device(device, non_blocking=True)
        transfer_end.record(transfer_stream)
        transfer_tracker.event_pairs.append((transfer_start, transfer_end))
    while True:
        try:
            next_cpu_batch = next(cpu_batches)
        except StopIteration:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
            yield current_device_batch
            return
        with torch.cuda.stream(transfer_stream):
            transfer_start = torch.cuda.Event(enable_timing=True)
            transfer_end = torch.cuda.Event(enable_timing=True)
            transfer_start.record(transfer_stream)
            next_device_batch = next_cpu_batch.to_device(device, non_blocking=True)
            transfer_end.record(transfer_stream)
            transfer_tracker.event_pairs.append((transfer_start, transfer_end))
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
        yield current_device_batch
        current_device_batch = next_device_batch


class Trainer:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingParams,
        objective: TrainingObjective,
        training_model: nn.Module | None = None,
        rank: int = 0,
    ) -> None:
        self.model: Network = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.args: TrainingParams = args
        self.training_model = _LogitForward(model) if training_model is None else training_model
        self.rank = rank
        self.objective = objective
        self.last_transfer_seconds = 0.0

    def _calculate_loss_for_batch(self, batch: TrainingBatch) -> LossResult:
        return self.objective.calculate_loss(self.training_model, batch, self.model.device)

    def _train_epoch(
        self,
        dataloader: TrainingBatchLoader,
        optimizer_step: int = 0,
    ) -> TrainingStats:
        self.model.train()
        self.training_model.train()
        termination_offset = 1
        ply_offset = termination_offset + len(TerminationReason)
        material_offset = ply_offset + len(PLY_VALUE_BIN_LABELS)
        value_metric_group_count = material_offset + len(MATERIAL_VALUE_BIN_LABELS)
        reduction_width = BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * value_metric_group_count
        reduction_values = torch.zeros(reduction_width, device=self.model.device, dtype=torch.float64)
        scaler = GradScaler(self.model.device.type, enabled=self.model.device.type == 'cuda')
        policy_losses: list[torch.Tensor] = []
        policy_loss_sample_counts: list[int] = []
        gradient_norms: list[torch.Tensor] = []
        metric_batches: list[_DetachedMetricBatch] = []

        transfer_tracker = DeviceTransferTracker(event_pairs=[])
        for batch in prefetch_device_training_batches(dataloader, self.model.device, transfer_tracker):
            self.optimizer.zero_grad()
            sample_count = batch.states.shape[0]

            with autocast(self.model.device.type, dtype=torch.bfloat16):
                loss_result = self.objective.calculate_loss(
                    self.training_model,
                    batch,
                    self.model.device,
                )

            scaler.scale(loss_result.total_loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            policy_losses.append(loss_result.policy_loss.detach())
            policy_loss_sample_counts.append(sample_count)
            if self.rank == 0:
                gradient_norms.append(grad_norm.detach())
            metric_batches.append(self._detach_metric_batch(loss_result, batch))

        if not metric_batches:
            raise ValueError('Training requires at least one batch.')
        self.last_transfer_seconds = transfer_tracker.elapsed_seconds()
        metric_batch = self._concatenate_metric_batches(metric_batches)
        metrics = self._calculate_value_metric_tensors(metric_batch)
        sample_count = sum(policy_loss_sample_counts)
        policy_loss_counts = torch.tensor(
            policy_loss_sample_counts,
            dtype=policy_losses[0].dtype,
            device=self.model.device,
        )
        reduction_values[0] = (torch.stack(policy_losses) * policy_loss_counts).double().sum()
        reduction_values[1] = sample_count
        reduction_values[2] = metrics.expected_scores.double().sum()
        reduction_values[3] = metrics.expected_scores.double().square().sum()
        if self.rank == 0:
            reduction_values[4] = torch.stack(gradient_norms).double().sum()
            reduction_values[5] = len(gradient_norms)
            reduction_values[6] = len(policy_losses)
        reduction_values[7] = metrics.value_loss_contributions.double().sum()

        metric_inputs = self._value_metric_inputs(metrics)
        self._accumulate_value_metrics(
            reduction_values,
            BASE_REDUCTION_WIDTH,
            metrics,
            metric_inputs,
            torch.ones(sample_count, dtype=torch.bool, device=self.model.device),
        )
        for reason in TerminationReason:
            self._accumulate_value_metrics(
                reduction_values,
                BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (1 + int(reason)),
                metrics,
                metric_inputs,
                metric_inputs.termination_reasons.eq(int(reason)),
            )
        sampled_metric_batch = self._concatenate_metric_batches(metric_batches[::SLICED_VALUE_METRIC_BATCH_INTERVAL])
        sampled_metrics = self._calculate_value_metric_tensors(sampled_metric_batch)
        sampled_metric_inputs = self._value_metric_inputs(sampled_metrics)
        for bin_index, sample_mask in enumerate(_fixed_bin_masks(sampled_metrics.plies, PLY_VALUE_BIN_UPPER_BOUNDS)):
            self._accumulate_value_metrics(
                reduction_values,
                BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (ply_offset + bin_index),
                sampled_metrics,
                sampled_metric_inputs,
                sample_mask,
            )
        material_counts = sampled_metrics.current_player_piece_counts + sampled_metrics.opponent_piece_counts
        for bin_index, sample_mask in enumerate(
            _fixed_bin_masks(material_counts, MATERIAL_VALUE_BIN_UPPER_BOUNDS, inclusive=True)
        ):
            self._accumulate_value_metrics(
                reduction_values,
                BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (material_offset + bin_index),
                sampled_metrics,
                sampled_metric_inputs,
                sample_mask,
            )

        if distributed.is_initialized():
            distributed.all_reduce(reduction_values, op=distributed.ReduceOp.SUM)

        return TrainingStats(
            policy_loss_sum=float(reduction_values[0].item()),
            value_loss_sum=float(reduction_values[7].item()),
            sample_count=int(reduction_values[1].item()),
            value_metrics=_value_metrics_from_reduction(reduction_values, BASE_REDUCTION_WIDTH),
            termination_value_metrics=tuple(
                _value_metrics_from_reduction(
                    reduction_values,
                    BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (1 + int(reason)),
                )
                for reason in TerminationReason
            ),
            value_sum=float(reduction_values[2].item()),
            value_square_sum=float(reduction_values[3].item()),
            gradient_norm_sum=float(reduction_values[4].item()),
            gradient_norm_count=int(reduction_values[5].item()),
            num_batches=int(reduction_values[6].item()),
            mcts_value_target_weight=self.objective.value_target_weight(optimizer_step),
            policy_loss_weight=self.objective.policy_loss_weight,
            value_loss_weight=self.objective.value_loss_weight,
            ply_value_metrics=tuple(
                _value_metrics_from_reduction(
                    reduction_values,
                    BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (ply_offset + bin_index),
                )
                for bin_index in range(len(PLY_VALUE_BIN_LABELS))
            ),
            material_value_metrics=tuple(
                _value_metrics_from_reduction(
                    reduction_values,
                    BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (material_offset + bin_index),
                )
                for bin_index in range(len(MATERIAL_VALUE_BIN_LABELS))
            ),
        )

    def _detach_metric_batch(
        self,
        loss_result: LossResult,
        batch: TrainingBatch,
    ) -> _DetachedMetricBatch:
        return _DetachedMetricBatch(
            value_logits=loss_result.value_logits.detach(),
            target_expected_scores=loss_result.target_expected_scores.detach(),
            value_loss_contributions=loss_result.value_loss_contributions.detach(),
            outcome_target_eligible=loss_result.outcome_target_eligible,
            material_target_eligible=loss_result.material_target_eligible,
            termination_reasons=batch.termination_reasons.to(device=self.model.device),
            final_outcomes=loss_result.final_outcomes,
            mcts_root_values=loss_result.mcts_root_values,
            material_result_scores=loss_result.material_result_scores,
            plies=batch.plies.to(device=self.model.device),
            current_player_piece_counts=batch.current_player_piece_counts.to(device=self.model.device),
            opponent_piece_counts=batch.opponent_piece_counts.to(device=self.model.device),
        )

    def _concatenate_metric_batches(
        self,
        batches: list[_DetachedMetricBatch],
    ) -> _DetachedMetricBatch:
        return _DetachedMetricBatch(
            value_logits=torch.cat(tuple(batch.value_logits for batch in batches)),
            target_expected_scores=torch.cat(tuple(batch.target_expected_scores for batch in batches)),
            value_loss_contributions=torch.cat(tuple(batch.value_loss_contributions for batch in batches)),
            outcome_target_eligible=torch.cat(tuple(batch.outcome_target_eligible for batch in batches)),
            material_target_eligible=torch.cat(tuple(batch.material_target_eligible for batch in batches)),
            termination_reasons=torch.cat(tuple(batch.termination_reasons for batch in batches)),
            final_outcomes=torch.cat(tuple(batch.final_outcomes for batch in batches)),
            mcts_root_values=torch.cat(tuple(batch.mcts_root_values for batch in batches)),
            material_result_scores=torch.cat(tuple(batch.material_result_scores for batch in batches)),
            plies=torch.cat(tuple(batch.plies for batch in batches)),
            current_player_piece_counts=torch.cat(tuple(batch.current_player_piece_counts for batch in batches)),
            opponent_piece_counts=torch.cat(tuple(batch.opponent_piece_counts for batch in batches)),
        )

    def _calculate_value_metric_tensors(
        self,
        metrics: _DetachedMetricBatch,
    ) -> _ValueMetricTensors:
        value_probabilities = torch.softmax(metrics.value_logits, dim=1)
        expected_scores = wdl_to_scalar(value_probabilities)
        return _ValueMetricTensors(
            value_probabilities=value_probabilities,
            expected_scores=expected_scores,
            target_expected_scores=metrics.target_expected_scores,
            value_loss_contributions=metrics.value_loss_contributions,
            outcome_losses=F.cross_entropy(metrics.value_logits, metrics.final_outcomes, reduction='none'),
            mcts_huber_losses=F.huber_loss(
                expected_scores,
                metrics.mcts_root_values,
                reduction='none',
            ),
            material_huber_losses=F.huber_loss(
                expected_scores,
                metrics.material_result_scores,
                reduction='none',
            ),
            outcome_target_eligible=metrics.outcome_target_eligible,
            material_target_eligible=metrics.material_target_eligible,
            termination_reasons=metrics.termination_reasons,
            final_outcomes=metrics.final_outcomes,
            material_result_scores=metrics.material_result_scores,
            plies=metrics.plies,
            current_player_piece_counts=metrics.current_player_piece_counts,
            opponent_piece_counts=metrics.opponent_piece_counts,
        )

    def _value_metric_inputs(
        self,
        metrics: _ValueMetricTensors,
    ) -> _ValueMetricInputs:
        expected_score_errors = metrics.expected_scores - metrics.target_expected_scores
        final_outcome_probabilities = F.one_hot(
            metrics.final_outcomes,
            num_classes=len(FinalOutcome),
        ).to(dtype=metrics.value_probabilities.dtype)
        return _ValueMetricInputs(
            outcome_target_eligible=metrics.outcome_target_eligible,
            mcts_target_eligible=metrics.termination_reasons.ne(int(TerminationReason.DIAGNOSTIC)),
            termination_reasons=metrics.termination_reasons,
            final_outcomes=metrics.final_outcomes,
            predicted_classes=metrics.value_probabilities.argmax(dim=1),
            target_expected_scores=metrics.target_expected_scores.to(dtype=torch.float64),
            calibration_bin_indices=torch.clamp(
                ((metrics.expected_scores + 1.0) * (EXPECTED_SCORE_CALIBRATION_BINS / 2.0)).to(dtype=torch.int64),
                min=0,
                max=EXPECTED_SCORE_CALIBRATION_BINS - 1,
            ),
            brier_scores=torch.square(metrics.value_probabilities - final_outcome_probabilities).sum(dim=1),
            expected_score_squared_errors=torch.square(expected_score_errors),
            expected_score_absolute_errors=torch.abs(expected_score_errors),
        )

    def _accumulate_value_metrics(
        self,
        reduction_values: torch.Tensor,
        offset: int,
        metrics: _ValueMetricTensors,
        metric_inputs: _ValueMetricInputs,
        sample_mask: torch.Tensor,
    ) -> None:
        outcome_mask = sample_mask & metric_inputs.outcome_target_eligible
        mcts_mask = sample_mask & metric_inputs.mcts_target_eligible
        material_mask = sample_mask & metrics.material_target_eligible

        reduction_values[offset] += metrics.outcome_losses[outcome_mask].double().sum()
        reduction_values[offset + 1] += metric_inputs.brier_scores[outcome_mask].double().sum()
        reduction_values[offset + 2] += metric_inputs.expected_score_squared_errors[outcome_mask].double().sum()
        reduction_values[offset + 3] += metric_inputs.expected_score_absolute_errors[outcome_mask].double().sum()
        reduction_values[offset + 4] += metrics.expected_scores[outcome_mask].double().sum()
        reduction_values[offset + 5] += metric_inputs.target_expected_scores[outcome_mask].sum()
        reduction_values[offset + 6] += outcome_mask.sum()
        for outcome in FinalOutcome:
            class_index = int(outcome)
            class_mask = outcome_mask & metric_inputs.final_outcomes.eq(class_index)
            reduction_values[offset + 11 + class_index] += (
                metrics.value_probabilities[class_mask, class_index].double().sum()
            )
            reduction_values[offset + 14 + class_index] += (
                metric_inputs.predicted_classes[class_mask].eq(class_index).sum()
            )
            reduction_values[offset + 17 + class_index] += class_mask.sum()
        eligible_bin_indices = metric_inputs.calibration_bin_indices[outcome_mask]
        bin_prediction_sums = torch.zeros(
            EXPECTED_SCORE_CALIBRATION_BINS,
            dtype=torch.float64,
            device=self.model.device,
        ).scatter_add_(
            0,
            eligible_bin_indices,
            metrics.expected_scores[outcome_mask].double(),
        )
        bin_target_sums = torch.zeros_like(bin_prediction_sums).scatter_add_(
            0,
            eligible_bin_indices,
            metric_inputs.target_expected_scores[outcome_mask],
        )
        bin_counts = torch.bincount(
            eligible_bin_indices,
            minlength=EXPECTED_SCORE_CALIBRATION_BINS,
        )
        reduction_values[offset + 20 : offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS] += bin_prediction_sums
        reduction_values[
            offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS : offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS * 2
        ] += bin_target_sums
        reduction_values[
            offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS * 2 : offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS * 3
        ] += bin_counts

        reduction_values[offset + 7] += metrics.mcts_huber_losses[mcts_mask].double().sum()
        reduction_values[offset + 8] += mcts_mask.sum()
        reduction_values[offset + 9] += metrics.material_huber_losses[material_mask].double().sum()
        reduction_values[offset + 10] += material_mask.sum()

    @timeit
    def train(
        self,
        dataloader: TrainingBatchLoader,
        optimizer_step: int,
    ) -> TrainingStats:
        """Train policy and the blended soft-WDL value target."""
        base_lr: float = self.args.learning_rate(optimizer_step, self.args.optimizer)
        if self.rank == 0:
            log_scalar('training/learning_rate', base_lr, optimizer_step)
            log(f'Setting learning rate to {base_lr} at optimizer step {optimizer_step}')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        return self._train_epoch(dataloader, optimizer_step)


def _fixed_bin_masks(
    values: torch.Tensor,
    upper_bounds: tuple[int, ...],
    inclusive: bool = False,
) -> tuple[torch.Tensor, ...]:
    masks: list[torch.Tensor] = []
    lower_bound: int | None = None
    for upper_bound in upper_bounds:
        upper_mask = values.le(upper_bound) if inclusive else values.lt(upper_bound)
        if lower_bound is not None:
            lower_mask = values.gt(lower_bound) if inclusive else values.ge(lower_bound)
            upper_mask &= lower_mask
        masks.append(upper_mask)
        lower_bound = upper_bound
    if lower_bound is None:
        return (torch.ones_like(values, dtype=torch.bool),)
    masks.append(values.gt(lower_bound) if inclusive else values.ge(lower_bound))
    return tuple(masks)


def _value_metrics_from_reduction(values: torch.Tensor, offset: int) -> ValueMetrics:
    return ValueMetrics(
        outcome_cross_entropy_sum=float(values[offset].item()),
        brier_score_sum=float(values[offset + 1].item()),
        expected_score_mse_sum=float(values[offset + 2].item()),
        expected_score_mae_sum=float(values[offset + 3].item()),
        predicted_expected_score_sum=float(values[offset + 4].item()),
        target_expected_score_sum=float(values[offset + 5].item()),
        outcome_target_count=int(values[offset + 6].item()),
        mcts_huber_sum=float(values[offset + 7].item()),
        mcts_target_count=int(values[offset + 8].item()),
        material_huber_sum=float(values[offset + 9].item()),
        material_target_count=int(values[offset + 10].item()),
        class_probability_sums=tuple(float(values[offset + 11 + index].item()) for index in range(3)),
        class_correct_counts=tuple(int(values[offset + 14 + index].item()) for index in range(3)),
        class_target_counts=tuple(int(values[offset + 17 + index].item()) for index in range(3)),
        expected_score_bin_prediction_sums=tuple(
            float(values[offset + 20 + index].item()) for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
        expected_score_bin_target_sums=tuple(
            float(values[offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS + index].item())
            for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
        expected_score_bin_counts=tuple(
            int(values[offset + 20 + EXPECTED_SCORE_CALIBRATION_BINS * 2 + index].item())
            for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
    )
