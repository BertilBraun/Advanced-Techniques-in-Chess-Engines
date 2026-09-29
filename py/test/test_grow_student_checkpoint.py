from __future__ import annotations

import math

import pytest
import torch
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.training.checkpoint.persistence import create_model
from src.training.network import (
    ChessFromToAttentionPolicyHeadConfiguration,
    GlobalPoolingResidualContext,
    NetworkParams,
    ResidualContextPlacement,
    ScaledPostActivationResidualBlockConfiguration,
)
from tools.grow_student_checkpoint import grow_student, grown_architecture, largest_output_difference


def small_student() -> NetworkParams:
    return NetworkParams(
        num_layers=4,
        hidden_size=32,
        residual_context=GlobalPoolingResidualContext(placement=ResidualContextPlacement.EVERY_SECOND_BLOCK),
        residual_block=ScaledPostActivationResidualBlockConfiguration(branch_scale=0.5, activation_cap=6.0),
        policy_head=ChessFromToAttentionPolicyHeadConfiguration(key_size=16),
        num_value_channels=2,
        value_fc_size=16,
    )


def random_states() -> torch.Tensor:
    dimensions = CHESS_NETWORK_DIMENSIONS
    shape = (16, dimensions.channels, dimensions.rows, dimensions.columns)
    return torch.randint(0, 2, shape, generator=torch.Generator().manual_seed(3)).float()


def test_grown_architecture_rescales_the_branch_to_the_new_depth() -> None:
    target = grown_architecture(small_student(), 6, 48)
    assert isinstance(target.residual_block, ScaledPostActivationResidualBlockConfiguration)
    assert target.residual_block.branch_scale == pytest.approx(1 / math.sqrt(6))


def test_grown_student_reproduces_its_source() -> None:
    torch.manual_seed(5)
    source = small_student()
    source_model = create_model(source, torch.device('cpu'), CHESS_NETWORK_DIMENSIONS)
    target = grown_architecture(source, 6, 48)
    target_model = grow_student(source_model, source, target, seed=7)
    assert largest_output_difference(source_model, target_model, random_states()) < 1e-4


def test_grown_architecture_rejects_a_narrower_target() -> None:
    with pytest.raises(ValueError):
        grown_architecture(small_student(), 6, 16)
