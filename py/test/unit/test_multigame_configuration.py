from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from src.az.config.resolution import resolve_configuration
from src.az.config.root import (
    ChessExperimentConfiguration,
    GoExperimentConfiguration,
    validate_resolved_configuration,
    validate_resolved_configuration_json,
)
from src.az.config.serialization import load_authoring_configuration
from src.az.games.api import GameIdentifier, create_game_registry
from src.az.games.chess.configuration import (
    ChessEvaluationConfiguration,
    ChessEvaluationSuite,
    ChessGameConfiguration,
    ChessModelConfiguration,
    ChessObjectiveConfiguration,
    ChessReplayConfiguration,
    ChessTrainingConfiguration,
    FixedChessModelSchedule,
    RandomChessOpponent,
    ResidualChessModelConfiguration,
    StockfishEngineConfiguration,
    StockfishNodesOpponent,
    StockfishSkillOpponent,
)


def go_configuration() -> GoExperimentConfiguration:
    authoring = load_authoring_configuration(
        Path("configs/go/go-7x7-fixed.authoring.json")
    )
    resolved = resolve_configuration(authoring)
    assert isinstance(resolved, GoExperimentConfiguration)
    return resolved


def stockfish_engine() -> StockfishEngineConfiguration:
    return StockfishEngineConfiguration(
        executable_path=PurePosixPath("engines/stockfish"),
        executable_sha256="1" * 64,
        protocol="uci",
        threads=1,
        hash_mebibytes=64,
        ponder=False,
    )


def chess_configuration() -> ChessExperimentConfiguration:
    go = go_configuration()
    return ChessExperimentConfiguration(
        schema_version=2,
        game="chess",
        experiment=go.experiment,
        hardware=go.hardware,
        topology=go.topology,
        game_configuration=ChessGameConfiguration(
            kind="chess",
            variant="standard",
            input_encoding="canonical_8x8_history_v1",
            action_encoding="az_8x8x73",
            history_length=8,
            repetition_draw_count=3,
            halfmove_draw_ply_count=100,
            insufficient_material_draw=True,
            safety_ply_cap=1_024,
            perspective="side_to_move",
            symmetry_group="horizontal_reflection",
        ),
        model=ChessModelConfiguration(
            schedule=FixedChessModelSchedule(
                kind="fixed",
                architecture=ResidualChessModelConfiguration(
                    family="residual_chess",
                    channels=128,
                    residual_blocks=10,
                    policy_channels=73,
                    value_hidden_size=256,
                    normalization="batch",
                    activation="relu",
                    value_head="wdl",
                ),
            )
        ),
        search=go.search,
        self_play=go.self_play,
        replay=ChessReplayConfiguration(
            kind="chess_layer_a_b",
            capacity_positions=2_500_000,
            layer_a_directory=PurePosixPath("runs/chess/layer-a"),
            layer_b_directory=PurePosixPath("runs/chess/layer-b"),
            maximum_layer_a_games_per_shard=32,
            maximum_layer_b_positions_per_shard=16_384,
            layer_a_schema_version=1,
            layer_b_schema_version=1,
            compression="none",
            sampling="uniform",
            credits=go.replay.credits,
        ),
        training=ChessTrainingConfiguration(
            global_batch_size=go.training.global_batch_size,
            local_batch_size=go.training.local_batch_size,
            maximum_optimizer_steps=go.training.maximum_optimizer_steps,
            optimizer=go.training.optimizer,
            learning_rate_schedule=go.training.learning_rate_schedule,
            precision=go.training.precision,
            objective=ChessObjectiveConfiguration(
                kind="chess_policy_wdl",
                policy_loss_weight=1,
                value_loss_weight=1,
                l2_regularization_weight=0,
            ),
            checkpoint_every_optimizer_steps=go.training.checkpoint_every_optimizer_steps,
            gradient_clip_norm=go.training.gradient_clip_norm,
        ),
        evaluation=ChessEvaluationConfiguration(
            search=go.evaluation.search,
            checkpoint_elapsed_seconds=go.evaluation.checkpoint_elapsed_seconds,
            paired_games_per_checkpoint=go.evaluation.paired_games_per_checkpoint,
            bootstrap_samples=go.evaluation.bootstrap_samples,
            confidence_method="paired_bootstrap",
            confidence_level=go.evaluation.confidence_level,
            bootstrap_seed=go.evaluation.bootstrap_seed,
            suite=ChessEvaluationSuite(
                kind="chess_ladder",
                opponents=(
                    RandomChessOpponent(kind="random"),
                    StockfishSkillOpponent(
                        kind="stockfish_skill",
                        engine=stockfish_engine(),
                        skill_level=3,
                    ),
                    StockfishNodesOpponent(
                        kind="stockfish_nodes",
                        engine=stockfish_engine(),
                        nodes_per_move=1_000,
                    ),
                ),
                alternate_colors=True,
                opening_source="initial_position",
            ),
        ),
        telemetry=go.telemetry,
        retention=go.retention,
    )


def test_root_discriminator_round_trips_complete_game_branches() -> None:
    go = go_configuration()
    chess = chess_configuration()

    assert validate_resolved_configuration_json(go.model_dump_json()) == go
    assert validate_resolved_configuration_json(chess.model_dump_json()) == chess
    assert chess.game_configuration.action_count == 4_672


@pytest.mark.parametrize(
    "field_name", ("game_configuration", "model", "replay", "training", "evaluation")
)
def test_chess_branch_rejects_go_specific_components(field_name: str) -> None:
    chess = chess_configuration().model_dump(mode="python")
    go = go_configuration().model_dump(mode="python")
    chess[field_name] = go[field_name]

    with pytest.raises(ValidationError):
        validate_resolved_configuration(chess)


def test_go_branch_rejects_chess_stockfish_evaluation() -> None:
    go = go_configuration().model_dump(mode="python")
    go["evaluation"] = chess_configuration().evaluation.model_dump(mode="python")

    with pytest.raises(ValidationError):
        validate_resolved_configuration(go)


def test_chess_identifier_exists_without_fake_runtime_registration() -> None:
    assert GameIdentifier.CHESS.value == "chess"
    with pytest.raises(ValueError, match="not available yet"):
        create_game_registry().resolve(GameIdentifier.CHESS)
