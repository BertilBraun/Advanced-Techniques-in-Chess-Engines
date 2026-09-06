from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import Field
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.objective import ResolvedTrainingObjective, blended_wdl_targets, mask_policy_logits
from src.util.frozen_model import FrozenModel
from torch.nn import functional


class PolicyTrainingDistribution(FrozenModel):
    loss: tuple[float, ...]
    target_top1_mass: tuple[float, ...]
    target_top2_mass: tuple[float, ...]
    target_top3_mass: tuple[float, ...]
    target_entropy: tuple[float, ...]
    prediction_entropy: tuple[float, ...]


class NextPolicyTrainingDistribution(FrozenModel):
    kind: Literal['next_policy'] = 'next_policy'
    policy: PolicyTrainingDistribution


class RemainingGameLengthTrainingDistribution(FrozenModel):
    kind: Literal['remaining_game_length'] = 'remaining_game_length'
    target: tuple[float, ...]
    prediction: tuple[float, ...]
    absolute_error: tuple[float, ...]


class ScalarAuxiliaryTrainingDistribution(FrozenModel):
    kind: Literal['future_search_value', 'irreversible_progress']
    target: tuple[float, ...]
    prediction: tuple[float, ...]
    absolute_error: tuple[float, ...]


class LegalMovesTrainingDistribution(FrozenModel):
    kind: Literal['legal_moves'] = 'legal_moves'
    legal_probability: tuple[float, ...]
    illegal_probability: tuple[float, ...]


AuxiliaryTrainingDistribution: TypeAlias = Annotated[
    NextPolicyTrainingDistribution
    | RemainingGameLengthTrainingDistribution
    | ScalarAuxiliaryTrainingDistribution
    | LegalMovesTrainingDistribution,
    Field(discriminator='kind'),
]


class TrainingDistributionSnapshot(FrozenModel):
    policy: PolicyTrainingDistribution
    wdl_loss: tuple[float, ...]
    root_value: tuple[float, ...]
    terminal_value: tuple[float, ...]
    predicted_value: tuple[float, ...]
    value_absolute_error: tuple[float, ...]
    sample_weight: tuple[float, ...]
    replay_generation_age: tuple[float, ...]
    replay_age_seconds: tuple[float, ...]
    auxiliary: tuple[AuxiliaryTrainingDistribution, ...]


def capture_training_distributions(
    output: TrainingModelOutput,
    batch: TrainingBatch,
    objective: ResolvedTrainingObjective,
    source_generation: int,
    captured_at_seconds: float,
    maximum_rows: int = 256,
) -> TrainingDistributionSnapshot:
    if maximum_rows <= 0:
        raise ValueError('Training distribution sample size must be positive.')
    row_count = min(len(batch), maximum_rows)
    rows = slice(0, row_count)
    policy_logits = output.policy_logits[rows].float()
    policy_targets = batch.policy_targets[rows]
    wdl_logits = output.wdl_logits[rows].float()
    wdl_targets = batch.wdl_targets[rows]
    root_values = batch.root_values[rows]
    policy = _policy_distribution(policy_logits, policy_targets, batch.policy_legal_action_ids[rows])
    blended_wdl = blended_wdl_targets(wdl_targets, root_values, objective.root_value_blend)
    wdl_loss = functional.cross_entropy(wdl_logits, blended_wdl, reduction='none')
    predicted_wdl = torch.softmax(wdl_logits, dim=1)
    value_coefficients = torch.tensor((1.0, 0.0, -1.0), device=predicted_wdl.device)
    predicted_values = predicted_wdl @ value_coefficients
    terminal_values = wdl_targets @ value_coefficients
    generation_ages = source_generation - batch.source_model_generations[rows]
    replay_ages = captured_at_seconds - batch.source_created_at_seconds[rows]
    auxiliary = tuple(
        _auxiliary_distribution(
            prediction[rows].float(), target[rows], legal_actions[rows], eligibility[rows], configuration.kind
        )
        for prediction, target, legal_actions, eligibility, configuration in zip(
            output.auxiliary_logits,
            batch.auxiliary_targets,
            batch.auxiliary_legal_action_ids,
            batch.auxiliary_eligibility,
            objective.auxiliary_losses,
            strict=True,
        )
    )
    return TrainingDistributionSnapshot(
        policy=policy,
        wdl_loss=_floats(wdl_loss),
        root_value=_floats(root_values),
        terminal_value=_floats(terminal_values),
        predicted_value=_floats(predicted_values),
        value_absolute_error=_floats(torch.abs(predicted_values - terminal_values)),
        sample_weight=_floats(batch.sample_weights[rows]),
        replay_generation_age=_floats(generation_ages.clamp_min(0)),
        replay_age_seconds=_floats(replay_ages.clamp_min(0.0)),
        auxiliary=auxiliary,
    )


def _auxiliary_distribution(
    prediction: torch.Tensor,
    target: torch.Tensor,
    legal_action_ids: torch.Tensor,
    eligibility: torch.Tensor,
    kind: Literal[
        'next_policy',
        'remaining_game_length',
        'future_search_value',
        'irreversible_progress',
        'legal_moves',
    ],
) -> AuxiliaryTrainingDistribution:
    eligible = eligibility.to(dtype=torch.bool)
    match kind:
        case 'next_policy':
            return NextPolicyTrainingDistribution(
                policy=_policy_distribution(prediction[eligible], target[eligible], legal_action_ids[eligible])
            )
        case 'remaining_game_length':
            eligible_predictions = prediction[eligible].squeeze(1)
            eligible_targets = target[eligible].squeeze(1)
            return RemainingGameLengthTrainingDistribution(
                target=_floats(eligible_targets),
                prediction=_floats(eligible_predictions),
                absolute_error=_floats(torch.abs(eligible_predictions - eligible_targets)),
            )
        case 'future_search_value' | 'irreversible_progress':
            eligible_predictions = prediction[eligible].squeeze(1)
            if kind == 'irreversible_progress':
                eligible_predictions = torch.sigmoid(eligible_predictions)
            eligible_targets = target[eligible].squeeze(1)
            return ScalarAuxiliaryTrainingDistribution(
                kind=kind,
                target=_floats(eligible_targets),
                prediction=_floats(eligible_predictions),
                absolute_error=_floats(torch.abs(eligible_predictions - eligible_targets)),
            )
        case 'legal_moves':
            probabilities = torch.sigmoid(prediction[eligible])
            legal = target[eligible] > 0.5
            illegal = ~legal
            return LegalMovesTrainingDistribution(
                legal_probability=_floats((probabilities * legal).sum(dim=1) / legal.sum(dim=1).clamp_min(1)),
                illegal_probability=_floats((probabilities * illegal).sum(dim=1) / illegal.sum(dim=1).clamp_min(1)),
            )


def _policy_distribution(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_action_ids: torch.Tensor,
) -> PolicyTrainingDistribution:
    if not len(logits):
        return PolicyTrainingDistribution(
            loss=(),
            target_top1_mass=(),
            target_top2_mass=(),
            target_top3_mass=(),
            target_entropy=(),
            prediction_entropy=(),
        )
    masked_logits = mask_policy_logits(logits, legal_action_ids)
    target_masses = torch.sort(targets, dim=1, descending=True).values
    cumulative_masses = torch.cumsum(target_masses, dim=1)
    return PolicyTrainingDistribution(
        loss=_floats(functional.cross_entropy(masked_logits, targets, reduction='none')),
        target_top1_mass=_floats(cumulative_masses[:, 0]),
        target_top2_mass=_floats(cumulative_masses[:, min(1, cumulative_masses.shape[1] - 1)]),
        target_top3_mass=_floats(cumulative_masses[:, min(2, cumulative_masses.shape[1] - 1)]),
        target_entropy=_floats(_entropy(targets)),
        prediction_entropy=_floats(_entropy(torch.softmax(masked_logits, dim=1))),
    )


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=1)


def _floats(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu())
