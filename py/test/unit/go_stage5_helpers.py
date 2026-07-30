from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from src.az.games.api import GameIdentifier
from src.az.games.go.configuration import (
    DisabledResignation,
    GoGameConfiguration,
    GoObjectiveConfiguration,
    ResidualGoModelConfiguration,
)
from src.az.games.go.samples import DensePolicyTarget, GoSample
from src.az.replay.envelope import (
    GameTermination,
    NoSearchCalibration,
    ReplayEnvelope,
    RootDiagnostics,
    SearchBudgetClass,
    SearchStopReason,
    SearchStrategy,
    derive_self_play_seed_lineage,
)


def game_configuration(board_size: int = 7, history_length: int = 2) -> GoGameConfiguration:
    return GoGameConfiguration(
        kind='go',
        board_size=board_size,
        komi_half_points=15,
        scoring_rule='area',
        ko_rule='positional_superko',
        suicide_rule='illegal',
        pass_exempt_from_superko=True,
        score_comparison='doubled_integer_points',
        safety_ply_cap=board_size**2 * 3,
        history_length=history_length,
        history_planes_per_position=2,
        include_color_plane=True,
        pass_action='last',
        normal_termination='two_consecutive_passes',
        symmetry_group='dihedral_8',
        capped_game_value_target_weight=0,
        resignation=DisabledResignation(kind='disabled'),
    )


def model_configuration() -> ResidualGoModelConfiguration:
    return ResidualGoModelConfiguration(
        family='residual_go',
        channels=8,
        residual_blocks=2,
        policy_channels=2,
        value_hidden_size=16,
        normalization='batch',
        activation='relu',
    )


def objective_configuration(l2_weight: float = 0.0) -> GoObjectiveConfiguration:
    return GoObjectiveConfiguration(
        kind='go_policy_value',
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        l2_regularization_weight=l2_weight,
    )


def sample(board_size: int = 7, history_length: int = 2, value_target: float | None = 1.0) -> GoSample:
    configuration = game_configuration(board_size, history_length)
    inputs = (
        np.arange(
            configuration.input_plane_count * board_size**2,
            dtype=np.int32,
        )
        % 2
    ).astype(np.float32)
    inputs = inputs.reshape(configuration.input_plane_count, board_size, board_size)
    legal = np.ones(configuration.action_count, dtype=np.bool_)
    policy = np.arange(1, configuration.action_count + 1, dtype=np.float32)
    return GoSample(
        input_planes=inputs,
        legal_action_mask=legal,
        policy_target=DensePolicyTarget(policy),
        policy_weight=1.0,
        value_target=value_target,
        value_weight=0.0 if value_target is None else 1.0,
    )


def envelope(
    sample_index: int = 1,
    termination: GameTermination = GameTermination.TWO_CONSECUTIVE_PASSES,
) -> ReplayEnvelope:
    root_seed = 123
    return ReplayEnvelope(
        run_id=UUID(int=1),
        game_identifier=GameIdentifier.GO,
        payload_schema_version=1,
        sample_id=UUID(int=sample_index + 10),
        game_id=UUID(int=2),
        seed_lineage=derive_self_play_seed_lineage(
            root_seed=root_seed,
            process_index=0,
            worker_index=0,
            game_index=0,
            ply=sample_index,
        ),
        created_at=datetime(2026, 7, 30, 0, sample_index, tzinfo=timezone.utc),
        ply=sample_index,
        checkpoint_id='checkpoint-000001',
        search_strategy=SearchStrategy.FIXED,
        budget_class=SearchBudgetClass.FIXED,
        configured_simulation_cap=16,
        actual_simulations=16,
        stop_reason=SearchStopReason.FULL_BUDGET,
        policy_target_eligible=True,
        policy_target_weight=1.0,
        value_target_eligible=termination is not GameTermination.SAFETY_PLY_CAP,
        value_target_weight=0.0 if termination is GameTermination.SAFETY_PLY_CAP else 1.0,
        root_diagnostics=RootDiagnostics(
            visit_count=16,
            entropy=1.5,
            top_two_margin=0.25,
            prefix_full_policy_disagreement=None,
            prefix_full_value_disagreement=None,
        ),
        termination=termination,
        replay_credit_id=UUID(int=20 + sample_index),
        search_calibration=NoSearchCalibration(kind='none'),
    )
