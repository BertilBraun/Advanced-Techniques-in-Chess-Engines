import torch
import torch.nn.functional as functional
from torch import nn

from src.neural_network import NetworkDimensions
from src.games.chess.configuration import (
    ChessExperimentConfiguration,
    ChessTrainingObjectiveConfiguration,
)
from src.games.training_contract import GameImplementation
from src.games.chess.self_play import (
    ChessSelfPlayPolicy,
    SelfPlayGame,
    SelfPlayStatisticsSnapshot,
)
from src.games.chess.completed_game import ChessCompletedGame
from src.self_play.completed_game import CompletedGamePublisher
from src.games.chess.replay import CHESS_REPLAY_IMPLEMENTATION
from src.training.replay import ReplayGameImplementation
from src.training.batch import TrainingBatch
from src.training.configuration import TrainingParams
from src.training.trainer import LossResult, TrainingObjective
from src.self_play.value_target import FinalOutcome
from src.value import scalar_to_wdl


class ChessTrainingObjective(TrainingObjective):
    def __init__(
        self,
        parameters: TrainingParams,
        configuration: ChessTrainingObjectiveConfiguration,
        optimizer_step: int,
        value_target_weight_override: float | None = None,
    ) -> None:
        self.parameters = parameters
        self.configuration = configuration
        self.optimizer_step = optimizer_step
        self.value_target_weight_override = value_target_weight_override

    @property
    def policy_loss_weight(self) -> float:
        return self.configuration.policy_loss_weight

    @property
    def value_loss_weight(self) -> float:
        return self.configuration.value_loss_weight

    def value_target_weight(self, optimizer_step: int) -> float:
        if self.value_target_weight_override is not None:
            return self.value_target_weight_override
        warmup_steps = self.configuration.mcts_value_target_warmup_optimizer_steps
        warmup_progress = 1.0 if warmup_steps == 0 else min(1.0, optimizer_step / warmup_steps)
        return self.configuration.mcts_value_loss_weight * warmup_progress

    def calculate_loss(
        self,
        training_model: nn.Module,
        batch: TrainingBatch,
        device: torch.device,
    ) -> LossResult:
        mcts_value_target_weight = self.value_target_weight(self.optimizer_step)
        states = batch.states.to(device=device)
        policy_targets = batch.policy_targets.to(device=device)
        final_outcomes = batch.final_outcomes.to(device=device)
        mcts_root_values = batch.mcts_root_values.to(device=device)
        outcome_target_eligible = batch.outcome_target_eligible.to(device=device)
        material_result_scores = batch.material_result_scores.to(device=device)
        material_target_eligible = batch.material_target_eligible.to(device=device)
        sample_weights = batch.sample_weights.to(device=device)
        if self.parameters.duplicate_multiplicity_weight_cap is not None:
            sample_weights.clamp_(max=self.parameters.duplicate_multiplicity_weight_cap)
        sample_weights /= sample_weights.mean()
        policy_logits, value_logits = training_model(states)
        policy_losses = functional.cross_entropy(policy_logits, policy_targets, reduction='none')
        policy_loss = (policy_losses * sample_weights).mean()
        outcome_expected_scores = final_outcomes.eq(int(FinalOutcome.WIN)).to(
            dtype=value_logits.dtype
        ) - final_outcomes.eq(int(FinalOutcome.LOSS)).to(dtype=value_logits.dtype)
        base_target_eligible = torch.logical_or(outcome_target_eligible, material_target_eligible)
        base_expected_scores = torch.where(
            outcome_target_eligible,
            outcome_expected_scores,
            material_result_scores,
        )
        target_expected_scores = torch.lerp(
            base_expected_scores,
            mcts_root_values,
            mcts_value_target_weight,
        )
        value_losses = functional.cross_entropy(
            value_logits,
            scalar_to_wdl(target_expected_scores),
            reduction='none',
        )
        value_loss_contributions = value_losses * base_target_eligible.to(dtype=value_losses.dtype) * sample_weights
        value_loss = value_loss_contributions.sum() / value_losses.shape[0]
        total_loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
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


class ChessImplementation(
    GameImplementation[
        ChessCompletedGame,
        SelfPlayGame,
        'ChessSelfPlaySearchRequest',
        'ChessSelfPlaySearchResult',
        SelfPlayStatisticsSnapshot | None,
    ]
):
    def __init__(self, configuration: ChessExperimentConfiguration) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> ChessExperimentConfiguration:
        return self._configuration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return self.configuration.network_dimensions

    @property
    def replay(self) -> ReplayGameImplementation[ChessCompletedGame]:
        return CHESS_REPLAY_IMPLEMENTATION

    def objective(self, optimizer_step: int) -> TrainingObjective:
        return ChessTrainingObjective(
            self.training.trainer,
            self.configuration.chess.objective,
            optimizer_step,
        )

    def create_self_play_policy(
        self,
        device_id: int,
        publisher: CompletedGamePublisher,
    ) -> ChessSelfPlayPolicy:
        return ChessSelfPlayPolicy(
            device_id,
            self.configuration.chess.self_play,
            self.training.save_path,
            publisher,
        )
