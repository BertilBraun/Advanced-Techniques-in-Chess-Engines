from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast

from src.Network import Network
from src.self_play.SelfPlayDataset import TrainingBatch
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.settings import log_scalar
from src.train.TrainingArgs import TrainingParams
from src.train.TrainingStats import (
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


VALUE_METRIC_WIDTH = 18 + EXPECTED_SCORE_CALIBRATION_BINS * 3
BASE_REDUCTION_WIDTH = 7
SLICED_VALUE_METRIC_BATCH_INTERVAL = 10


@dataclass(frozen=True)
class _LossResult:
    policy_loss: torch.Tensor
    outcome_loss: torch.Tensor
    mcts_auxiliary_loss: torch.Tensor
    combined_value_loss: torch.Tensor
    total_loss: torch.Tensor
    value_probabilities: torch.Tensor
    expected_scores: torch.Tensor
    outcome_losses: torch.Tensor
    brier_scores: torch.Tensor
    expected_score_squared_errors: torch.Tensor
    expected_score_absolute_errors: torch.Tensor
    mcts_huber_losses: torch.Tensor


@dataclass(frozen=True)
class _ValueMetricInputs:
    outcome_target_eligible: torch.Tensor
    mcts_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor
    final_outcomes: torch.Tensor
    predicted_classes: torch.Tensor
    target_expected_scores: torch.Tensor
    calibration_bin_indices: torch.Tensor


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
        states = batch.states.to(device=self.model.device)
        policy_targets = batch.policy_targets.to(device=self.model.device)
        final_outcomes = batch.final_outcomes.to(device=self.model.device)
        mcts_root_values = batch.mcts_root_values.to(device=self.model.device)
        outcome_target_eligible = batch.outcome_target_eligible.to(device=self.model.device)
        termination_reasons = batch.termination_reasons.to(device=self.model.device)
        mcts_target_eligible = termination_reasons.ne(int(TerminationReason.DIAGNOSTIC))

        policy_logits, value_logits = self.training_model(states)
        value_probabilities = torch.softmax(value_logits, dim=1)
        expected_scores = wdl_to_scalar(value_probabilities)

        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        outcome_losses = F.cross_entropy(value_logits, final_outcomes, reduction='none')
        local_outcome_count = outcome_target_eligible.sum()
        global_outcome_count = local_outcome_count.detach().clone()
        if distributed.is_initialized():
            distributed.all_reduce(global_outcome_count, op=distributed.ReduceOp.SUM)
        if global_outcome_count.item() > 0:
            outcome_loss = outcome_losses[outcome_target_eligible].sum() / global_outcome_count
            if distributed.is_initialized():
                outcome_loss *= distributed.get_world_size()
        else:
            outcome_loss = value_logits.sum() * 0.0

        final_outcome_probabilities = F.one_hot(
            final_outcomes,
            num_classes=len(FinalOutcome),
        ).to(dtype=value_probabilities.dtype)
        brier_scores = torch.square(value_probabilities - final_outcome_probabilities).sum(dim=1)
        target_expected_scores = final_outcomes.eq(int(FinalOutcome.WIN)).to(
            dtype=value_probabilities.dtype
        ) - final_outcomes.eq(int(FinalOutcome.LOSS)).to(dtype=value_probabilities.dtype)
        expected_score_errors = expected_scores - target_expected_scores
        mcts_huber_losses = F.huber_loss(
            expected_scores,
            mcts_root_values,
            reduction='none',
        )
        local_mcts_count = mcts_target_eligible.sum()
        global_mcts_count = local_mcts_count.detach().clone()
        if distributed.is_initialized():
            distributed.all_reduce(global_mcts_count, op=distributed.ReduceOp.SUM)
        if global_mcts_count.item() > 0:
            mcts_auxiliary_loss = mcts_huber_losses[mcts_target_eligible].sum() / global_mcts_count
            if distributed.is_initialized():
                mcts_auxiliary_loss *= distributed.get_world_size()
        else:
            mcts_auxiliary_loss = value_logits.sum() * 0.0
        combined_value_loss = (
            self.args.outcome_value_loss_weight * outcome_loss + self.args.mcts_value_loss_weight * mcts_auxiliary_loss
        )
        total_loss = self.args.policy_loss_weight * policy_loss + self.args.value_loss_weight * combined_value_loss

        return _LossResult(
            policy_loss=policy_loss,
            outcome_loss=outcome_loss,
            mcts_auxiliary_loss=mcts_auxiliary_loss,
            combined_value_loss=combined_value_loss,
            total_loss=total_loss,
            value_probabilities=value_probabilities,
            expected_scores=expected_scores,
            outcome_losses=outcome_losses,
            brier_scores=brier_scores,
            expected_score_squared_errors=torch.square(expected_score_errors),
            expected_score_absolute_errors=torch.abs(expected_score_errors),
            mcts_huber_losses=mcts_huber_losses,
        )

    def _train_epoch(self, dataloader: TrainingBatchLoader) -> TrainingStats:
        self.model.train()
        self.training_model.train()
        termination_offset = 1
        ply_offset = termination_offset + len(TerminationReason)
        material_offset = ply_offset + len(PLY_VALUE_BIN_LABELS)
        value_metric_group_count = material_offset + len(MATERIAL_VALUE_BIN_LABELS)
        reduction_width = BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * value_metric_group_count
        reduction_values = torch.zeros(reduction_width, device=self.model.device, dtype=torch.float64)
        scaler = GradScaler(self.model.device.type, enabled=self.model.device.type == 'cuda')

        for batch_index, batch in enumerate(prefetch_training_batches(dataloader)):
            self.optimizer.zero_grad()
            sample_count = batch.states.shape[0]

            with autocast(self.model.device.type, dtype=torch.bfloat16):
                loss_result = self._calculate_loss_for_batch(batch)

            scaler.scale(loss_result.total_loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            reduction_values[0] += loss_result.policy_loss.detach().double() * sample_count
            reduction_values[1] += sample_count
            reduction_values[2] += loss_result.expected_scores.detach().double().sum()
            reduction_values[3] += loss_result.expected_scores.detach().double().square().sum()
            if self.rank == 0:
                reduction_values[4] += grad_norm.detach().double()
                reduction_values[5] += 1
                reduction_values[6] += 1

            metric_inputs = self._value_metric_inputs(loss_result, batch)
            self._accumulate_value_metrics(
                reduction_values,
                BASE_REDUCTION_WIDTH,
                loss_result,
                metric_inputs,
                torch.ones(sample_count, dtype=torch.bool, device=self.model.device),
            )
            for reason in TerminationReason:
                self._accumulate_value_metrics(
                    reduction_values,
                    BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (1 + int(reason)),
                    loss_result,
                    metric_inputs,
                    metric_inputs.termination_reasons.eq(int(reason)),
                )
            if batch_index % SLICED_VALUE_METRIC_BATCH_INTERVAL == 0:
                plies = batch.plies.to(device=self.model.device)
                for bin_index, sample_mask in enumerate(_fixed_bin_masks(plies, PLY_VALUE_BIN_UPPER_BOUNDS)):
                    self._accumulate_value_metrics(
                        reduction_values,
                        BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (ply_offset + bin_index),
                        loss_result,
                        metric_inputs,
                        sample_mask,
                    )
                material_counts = batch.current_player_piece_counts.to(
                    device=self.model.device
                ) + batch.opponent_piece_counts.to(device=self.model.device)
                for bin_index, sample_mask in enumerate(
                    _fixed_bin_masks(material_counts, MATERIAL_VALUE_BIN_UPPER_BOUNDS, inclusive=True)
                ):
                    self._accumulate_value_metrics(
                        reduction_values,
                        BASE_REDUCTION_WIDTH + VALUE_METRIC_WIDTH * (material_offset + bin_index),
                        loss_result,
                        metric_inputs,
                        sample_mask,
                    )

        if distributed.is_initialized():
            distributed.all_reduce(reduction_values, op=distributed.ReduceOp.SUM)

        return TrainingStats(
            policy_loss_sum=float(reduction_values[0].item()),
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
            outcome_value_loss_weight=self.args.outcome_value_loss_weight,
            mcts_value_loss_weight=self.args.mcts_value_loss_weight,
            policy_loss_weight=self.args.policy_loss_weight,
            value_loss_weight=self.args.value_loss_weight,
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

    def _value_metric_inputs(
        self,
        loss_result: _LossResult,
        batch: TrainingBatch,
    ) -> _ValueMetricInputs:
        final_outcomes = batch.final_outcomes.to(device=self.model.device)
        termination_reasons = batch.termination_reasons.to(device=self.model.device)
        expected_scores = loss_result.expected_scores
        return _ValueMetricInputs(
            outcome_target_eligible=batch.outcome_target_eligible.to(device=self.model.device),
            mcts_target_eligible=termination_reasons.ne(int(TerminationReason.DIAGNOSTIC)),
            termination_reasons=termination_reasons,
            final_outcomes=final_outcomes,
            predicted_classes=loss_result.value_probabilities.argmax(dim=1),
            target_expected_scores=final_outcomes.eq(int(FinalOutcome.WIN)).to(dtype=torch.float64)
            - final_outcomes.eq(int(FinalOutcome.LOSS)).to(dtype=torch.float64),
            calibration_bin_indices=torch.clamp(
                ((expected_scores + 1.0) * (EXPECTED_SCORE_CALIBRATION_BINS / 2.0)).to(dtype=torch.int64),
                min=0,
                max=EXPECTED_SCORE_CALIBRATION_BINS - 1,
            ),
        )

    def _accumulate_value_metrics(
        self,
        reduction_values: torch.Tensor,
        offset: int,
        loss_result: _LossResult,
        metric_inputs: _ValueMetricInputs,
        sample_mask: torch.Tensor,
    ) -> None:
        outcome_mask = sample_mask & metric_inputs.outcome_target_eligible
        mcts_mask = sample_mask & metric_inputs.mcts_target_eligible

        reduction_values[offset] += loss_result.outcome_losses[outcome_mask].detach().double().sum()
        reduction_values[offset + 1] += loss_result.brier_scores[outcome_mask].detach().double().sum()
        reduction_values[offset + 2] += loss_result.expected_score_squared_errors[outcome_mask].detach().double().sum()
        reduction_values[offset + 3] += loss_result.expected_score_absolute_errors[outcome_mask].detach().double().sum()
        reduction_values[offset + 4] += loss_result.expected_scores[outcome_mask].detach().double().sum()
        reduction_values[offset + 5] += metric_inputs.target_expected_scores[outcome_mask].sum()
        reduction_values[offset + 6] += outcome_mask.sum()
        for outcome in FinalOutcome:
            class_index = int(outcome)
            class_mask = outcome_mask & metric_inputs.final_outcomes.eq(class_index)
            reduction_values[offset + 9 + class_index] += (
                loss_result.value_probabilities[class_mask, class_index].detach().double().sum()
            )
            reduction_values[offset + 12 + class_index] += (
                metric_inputs.predicted_classes[class_mask].eq(class_index).sum()
            )
            reduction_values[offset + 15 + class_index] += class_mask.sum()
        eligible_bin_indices = metric_inputs.calibration_bin_indices[outcome_mask]
        bin_prediction_sums = torch.zeros(
            EXPECTED_SCORE_CALIBRATION_BINS,
            dtype=torch.float64,
            device=self.model.device,
        ).scatter_add_(
            0,
            eligible_bin_indices,
            loss_result.expected_scores[outcome_mask].detach().double(),
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
        reduction_values[offset + 18 : offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS] += bin_prediction_sums
        reduction_values[
            offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS : offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS * 2
        ] += bin_target_sums
        reduction_values[
            offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS * 2 : offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS * 3
        ] += bin_counts

        reduction_values[offset + 7] += loss_result.mcts_huber_losses[mcts_mask].detach().double().sum()
        reduction_values[offset + 8] += mcts_mask.sum()

    @timeit
    def train(
        self,
        dataloader: TrainingBatchLoader,
        iteration: int,
    ) -> TrainingStats:
        """Train the model with policy, hard WDL outcome, and MCTS auxiliary targets."""
        base_lr: float = self.args.learning_rate(iteration, self.args.optimizer)
        if self.rank == 0:
            log_scalar('training/learning_rate', base_lr, iteration)
            log(f'Setting learning rate to {base_lr} for iteration {iteration}')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

        return self._train_epoch(dataloader)


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
        class_probability_sums=tuple(float(values[offset + 9 + index].item()) for index in range(3)),
        class_correct_counts=tuple(int(values[offset + 12 + index].item()) for index in range(3)),
        class_target_counts=tuple(int(values[offset + 15 + index].item()) for index in range(3)),
        expected_score_bin_prediction_sums=tuple(
            float(values[offset + 18 + index].item()) for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
        expected_score_bin_target_sums=tuple(
            float(values[offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS + index].item())
            for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
        expected_score_bin_counts=tuple(
            int(values[offset + 18 + EXPECTED_SCORE_CALIBRATION_BINS * 2 + index].item())
            for index in range(EXPECTED_SCORE_CALIBRATION_BINS)
        ),
    )
