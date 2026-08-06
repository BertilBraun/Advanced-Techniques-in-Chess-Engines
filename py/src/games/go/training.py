from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from src.Network import Network
from src.experiment.chess_experiment import GoExperimentConfiguration, GoTrainingObjectiveConfiguration
from src.games.go.contract import GoStateContract
from src.self_play.SelfPlayDataset import TrainingBatch
from src.self_play.value_target import FinalOutcome
from src.value import scalar_to_wdl


@dataclass(frozen=True)
class GoLoss:
    policy: torch.Tensor
    value: torch.Tensor
    total: torch.Tensor


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
    policy_targets = batch.policy_targets.to(model.device)
    final_outcomes = batch.final_outcomes.to(model.device)
    root_values = batch.mcts_root_values.to(model.device)
    sample_weights = batch.sample_weights.to(model.device)
    sample_weights = sample_weights / sample_weights.mean()
    policy_logits, value_logits = model.logit_forward(states)
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
