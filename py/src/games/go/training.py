from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from src.neural_network import Network, NetworkDimensions
from src.games.training_contract import GameImplementation
from src.games.go.configuration import (
    GoExperimentConfiguration,
    GoTrainingObjectiveConfiguration,
)
from src.games.go.contract import GoStateContract
from src.games.go.completed_game import GoCompletedGame
from src.games.go.replay import GoReplayImplementation
from src.games.go.self_play import (
    GoSelfPlayGame,
    GoSelfPlayPolicy,
    NativeGoSearchRequest,
    NativeGoSearchResult,
)
from src.self_play.completed_game import CompletedGamePublisher
from src.training.batch import TrainingBatch
from src.training.replay import ReplayGameImplementation
from src.training.trainer import LossResult, TrainingObjective
from src.self_play.value_target import FinalOutcome
from src.value import scalar_to_wdl


@dataclass(frozen=True)
class GoLoss:
    policy: torch.Tensor
    value: torch.Tensor
    total: torch.Tensor


class GoTrainingObjective(TrainingObjective):
    def __init__(self, configuration: GoTrainingObjectiveConfiguration) -> None:
        self.configuration = configuration

    @property
    def policy_loss_weight(self) -> float:
        return self.configuration.policy_loss_weight

    @property
    def value_loss_weight(self) -> float:
        return 1.0

    def value_target_weight(self, optimizer_step: int) -> float:
        return self.configuration.root_value_loss_weight

    def calculate_loss(
        self,
        training_model: nn.Module,
        batch: TrainingBatch,
        device: torch.device,
    ) -> LossResult:
        states = batch.states.to(device=device)
        policy_targets = batch.policy_targets.to(device=device)
        final_outcomes = batch.final_outcomes.to(device=device)
        mcts_root_values = batch.mcts_root_values.to(device=device)
        outcome_target_eligible = batch.outcome_target_eligible.to(device=device)
        material_result_scores = batch.material_result_scores.to(device=device)
        material_target_eligible = batch.material_target_eligible.to(device=device)
        sample_weights = batch.sample_weights.to(device=device)
        sample_weights /= sample_weights.mean()
        policy_logits, value_logits = training_model(states)
        policy_rows = functional.cross_entropy(policy_logits, policy_targets, reduction='none')
        policy_loss = (policy_rows * sample_weights).mean()
        outcome_scores = final_outcomes.eq(int(FinalOutcome.WIN)).to(mcts_root_values.dtype) - final_outcomes.eq(
            int(FinalOutcome.LOSS)
        ).to(mcts_root_values.dtype)
        target_expected_scores = torch.lerp(
            outcome_scores,
            mcts_root_values,
            self.configuration.root_value_loss_weight,
        )
        value_rows = functional.cross_entropy(
            value_logits,
            scalar_to_wdl(target_expected_scores),
            reduction='none',
        )
        value_loss_contributions = value_rows * sample_weights
        value_loss = value_loss_contributions.mean()
        total_loss = self.policy_loss_weight * policy_loss + value_loss
        return LossResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            value_logits=value_logits,
            target_expected_scores=target_expected_scores,
            value_loss_contributions=value_loss_contributions,
            final_outcomes=final_outcomes,
            mcts_root_values=mcts_root_values,
            outcome_target_eligible=outcome_target_eligible,
            material_result_scores=material_result_scores,
            material_target_eligible=material_target_eligible,
        )


def create_go_model(configuration: GoExperimentConfiguration, device: torch.device) -> Network:
    contract = GoStateContract(
        configuration.go.representation.board_size,
        configuration.go.representation.history_length,
    )
    return Network(configuration.training.network, device, contract.network_dimensions)


def calculate_go_loss(
    model: Network,
    batch: TrainingBatch,
    objective: GoTrainingObjectiveConfiguration,
) -> GoLoss:
    states = batch.states.to(model.device)
    policy_logits, value_logits = model.logit_forward(states)
    return calculate_go_loss_from_logits(
        policy_logits,
        value_logits,
        batch,
        objective,
        model.device,
    )


def calculate_go_loss_from_logits(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    batch: TrainingBatch,
    objective: GoTrainingObjectiveConfiguration,
    device: torch.device,
) -> GoLoss:
    policy_targets = batch.policy_targets.to(device)
    final_outcomes = batch.final_outcomes.to(device)
    root_values = batch.mcts_root_values.to(device)
    sample_weights = batch.sample_weights.to(device)
    sample_weights = sample_weights / sample_weights.mean()
    policy_rows = functional.cross_entropy(policy_logits, policy_targets, reduction='none')
    policy_loss = (policy_rows * sample_weights).mean()
    outcome_scores = final_outcomes.eq(int(FinalOutcome.WIN)).to(value_logits.dtype) - final_outcomes.eq(
        int(FinalOutcome.LOSS)
    ).to(value_logits.dtype)
    target_scores = torch.lerp(outcome_scores, root_values, objective.root_value_loss_weight)
    value_targets = scalar_to_wdl(target_scores)
    value_rows = functional.cross_entropy(value_logits, value_targets, reduction='none')
    value_loss = (value_rows * sample_weights).mean()
    total = objective.policy_loss_weight * policy_loss + value_loss
    return GoLoss(policy=policy_loss, value=value_loss, total=total)


class GoImplementation(
    GameImplementation[
        GoCompletedGame,
        GoSelfPlayGame,
        NativeGoSearchRequest,
        NativeGoSearchResult,
        None,
    ]
):
    def __init__(self, configuration: GoExperimentConfiguration) -> None:
        self._configuration = configuration
        self.state = GoStateContract(
            configuration.go.representation.board_size,
            configuration.go.representation.history_length,
        )
        self._replay = GoReplayImplementation(self.state)

    @property
    def configuration(self) -> GoExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.configuration.network_dimensions

    @property
    def replay(self) -> ReplayGameImplementation[GoCompletedGame]:
        return self._replay

    def objective(self, optimizer_step: int) -> TrainingObjective:
        return GoTrainingObjective(self.configuration.go.objective)

    def create_self_play_policy(
        self,
        device_id: int,
        publisher: CompletedGamePublisher,
    ) -> GoSelfPlayPolicy:
        return GoSelfPlayPolicy(self.configuration, publisher, device_id)
