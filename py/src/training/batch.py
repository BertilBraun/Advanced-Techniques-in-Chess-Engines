from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingModelOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    auxiliary_logits: tuple[torch.Tensor, ...]
    features: torch.Tensor


@dataclass(frozen=True)
class TrainingBatch:
    states: torch.Tensor
    policy_targets: torch.Tensor
    policy_legal_action_ids: torch.Tensor
    wdl_targets: torch.Tensor
    root_values: torch.Tensor
    auxiliary_targets: tuple[torch.Tensor, ...]
    auxiliary_legal_action_ids: tuple[torch.Tensor, ...]
    auxiliary_eligibility: tuple[torch.Tensor, ...]
    sample_weights: torch.Tensor
    source_model_generations: torch.Tensor
    source_created_at_seconds: torch.Tensor

    def __post_init__(self) -> None:
        if not (len(self.auxiliary_targets) == len(self.auxiliary_legal_action_ids) == len(self.auxiliary_eligibility)):
            raise ValueError('Auxiliary targets, legal actions, and eligibility masks must have equal lengths.')

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def pin_memory(self) -> TrainingBatch:
        return TrainingBatch(
            states=self.states.pin_memory(),
            policy_targets=self.policy_targets.pin_memory(),
            policy_legal_action_ids=self.policy_legal_action_ids.pin_memory(),
            wdl_targets=self.wdl_targets.pin_memory(),
            root_values=self.root_values.pin_memory(),
            auxiliary_targets=tuple(target.pin_memory() for target in self.auxiliary_targets),
            auxiliary_legal_action_ids=tuple(actions.pin_memory() for actions in self.auxiliary_legal_action_ids),
            auxiliary_eligibility=tuple(mask.pin_memory() for mask in self.auxiliary_eligibility),
            sample_weights=self.sample_weights.pin_memory(),
            source_model_generations=self.source_model_generations.pin_memory(),
            source_created_at_seconds=self.source_created_at_seconds.pin_memory(),
        )

    def to_device(self, device: torch.device, non_blocking: bool) -> TrainingBatch:
        return TrainingBatch(
            states=self.states.to(device=device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device=device, non_blocking=non_blocking),
            policy_legal_action_ids=self.policy_legal_action_ids.to(device=device, non_blocking=non_blocking),
            wdl_targets=self.wdl_targets.to(device=device, non_blocking=non_blocking),
            root_values=self.root_values.to(device=device, non_blocking=non_blocking),
            auxiliary_targets=tuple(
                target.to(device=device, non_blocking=non_blocking) for target in self.auxiliary_targets
            ),
            auxiliary_legal_action_ids=tuple(
                actions.to(device=device, non_blocking=non_blocking) for actions in self.auxiliary_legal_action_ids
            ),
            auxiliary_eligibility=tuple(
                mask.to(device=device, non_blocking=non_blocking) for mask in self.auxiliary_eligibility
            ),
            sample_weights=self.sample_weights.to(device=device, non_blocking=non_blocking),
            source_model_generations=self.source_model_generations.to(device=device, non_blocking=non_blocking),
            source_created_at_seconds=self.source_created_at_seconds.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class RuntimeTrainingBatch:
    """Transitional batch consumed only by the pre-Phase-2 training runtime."""

    states: torch.Tensor
    policy_targets: torch.Tensor
    final_outcomes: torch.Tensor
    mcts_root_values: torch.Tensor
    outcome_target_eligible: torch.Tensor
    material_result_scores: torch.Tensor
    material_target_eligible: torch.Tensor
    termination_reasons: torch.Tensor
    plies: torch.Tensor
    current_player_piece_counts: torch.Tensor
    opponent_piece_counts: torch.Tensor
    occurrence_counts: torch.Tensor
    sample_weights: torch.Tensor

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def row_view(self, start: int, stop: int) -> RuntimeTrainingBatch:
        if not 0 <= start < stop <= len(self):
            raise ValueError(f'Batch row view [{start}, {stop}) is outside [0, {len(self)}).')
        rows = slice(start, stop)
        return RuntimeTrainingBatch(
            states=self.states[rows],
            policy_targets=self.policy_targets[rows],
            final_outcomes=self.final_outcomes[rows],
            mcts_root_values=self.mcts_root_values[rows],
            outcome_target_eligible=self.outcome_target_eligible[rows],
            material_result_scores=self.material_result_scores[rows],
            material_target_eligible=self.material_target_eligible[rows],
            termination_reasons=self.termination_reasons[rows],
            plies=self.plies[rows],
            current_player_piece_counts=self.current_player_piece_counts[rows],
            opponent_piece_counts=self.opponent_piece_counts[rows],
            occurrence_counts=self.occurrence_counts[rows],
            sample_weights=self.sample_weights[rows],
        )

    def pin_memory(self) -> RuntimeTrainingBatch:
        return RuntimeTrainingBatch(
            states=self.states.pin_memory(),
            policy_targets=self.policy_targets.pin_memory(),
            final_outcomes=self.final_outcomes.pin_memory(),
            mcts_root_values=self.mcts_root_values.pin_memory(),
            outcome_target_eligible=self.outcome_target_eligible.pin_memory(),
            material_result_scores=self.material_result_scores.pin_memory(),
            material_target_eligible=self.material_target_eligible.pin_memory(),
            termination_reasons=self.termination_reasons.pin_memory(),
            plies=self.plies.pin_memory(),
            current_player_piece_counts=self.current_player_piece_counts.pin_memory(),
            opponent_piece_counts=self.opponent_piece_counts.pin_memory(),
            occurrence_counts=self.occurrence_counts.pin_memory(),
            sample_weights=self.sample_weights.pin_memory(),
        )

    def to_device(self, device: torch.device, non_blocking: bool) -> RuntimeTrainingBatch:
        return RuntimeTrainingBatch(
            states=self.states.to(device=device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device=device, non_blocking=non_blocking),
            final_outcomes=self.final_outcomes.to(device=device, non_blocking=non_blocking),
            mcts_root_values=self.mcts_root_values.to(device=device, non_blocking=non_blocking),
            outcome_target_eligible=self.outcome_target_eligible.to(device=device, non_blocking=non_blocking),
            material_result_scores=self.material_result_scores.to(device=device, non_blocking=non_blocking),
            material_target_eligible=self.material_target_eligible.to(device=device, non_blocking=non_blocking),
            termination_reasons=self.termination_reasons.to(device=device, non_blocking=non_blocking),
            plies=self.plies.to(device=device, non_blocking=non_blocking),
            current_player_piece_counts=self.current_player_piece_counts.to(device=device, non_blocking=non_blocking),
            opponent_piece_counts=self.opponent_piece_counts.to(device=device, non_blocking=non_blocking),
            occurrence_counts=self.occurrence_counts.to(device=device, non_blocking=non_blocking),
            sample_weights=self.sample_weights.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ReplaySampleMetadata:
    ply: int
    current_player_piece_count: int
    opponent_piece_count: int
    occurrence_count: int = 1
    starting_fen: str | None = None
    moves_uci: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ply < 0:
            raise ValueError('Sample ply must be nonnegative.')
        if not 0 <= self.current_player_piece_count <= 127:
            raise ValueError('Current-player piece count must fit a signed int8.')
        if not 0 <= self.opponent_piece_count <= 127:
            raise ValueError('Opponent piece count must fit a signed int8.')
        if self.occurrence_count <= 0:
            raise ValueError('Replay occurrence count must be positive.')
        if self.moves_uci and self.starting_fen is None:
            raise ValueError('UCI move history requires a starting FEN.')
