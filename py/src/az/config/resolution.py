from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, PositiveInt, TypeAdapter

from src.az.config.base import FrozenModel
from src.az.config.evaluation import EvaluationConfiguration
from src.az.config.experiment import (
    ExperimentConfiguration,
    HardwareConfiguration,
    ManifestPolicy,
)
from src.az.config.model import ModelConfiguration
from src.az.config.profiles import (
    default_evaluation_opponent,
    default_evaluation_search,
    default_fpu,
    default_game,
    default_manifest_policy,
    default_model,
    default_replay_credits,
    default_retention,
    default_root_exploration,
    default_search_algorithm,
    default_search_inference,
    default_self_play,
    default_telemetry,
    default_temperature,
    default_topology,
    default_training,
    default_tree_reuse,
    planned_hardware_profile,
)
from src.az.config.root import (
    ChessExperimentConfiguration,
    GoExperimentConfiguration,
    ResolvedRunConfiguration,
)
from src.az.config.runtime import (
    RetentionConfiguration,
    TelemetryConfiguration,
    TopologyConfiguration,
)
from src.az.config.search import (
    FullBudgetStopping,
    FpuConfiguration,
    RootExplorationConfiguration,
    SearchAlgorithmConfiguration,
    SearchBudgetConfiguration,
    SearchConfiguration,
    SearchInferenceConfiguration,
    SearchStoppingConfiguration,
    TemperatureConfiguration,
    TreeReuseConfiguration,
)
from src.az.config.training import (
    ReplayConfiguration,
    SelfPlayConfiguration,
    TrainingConfiguration,
)
from src.az.games.go.configuration import (
    GoEvaluationSuite,
    GoGameConfiguration,
    GoOpponentConfiguration,
)
from src.az.games.chess.configuration import (
    ChessEvaluationConfiguration,
    ChessGameConfiguration,
    ChessModelConfiguration,
    ChessReplayConfiguration,
    ChessTrainingConfiguration,
)


class AuthoringExperimentConfiguration(ExperimentConfiguration):
    duration_seconds: PositiveInt = 21_600
    checkpoint_elapsed_seconds: tuple[PositiveInt, ...] = (
        900,
        1_800,
        3_600,
        7_200,
        14_400,
        21_600,
    )
    manifest_policy: ManifestPolicy = Field(default_factory=default_manifest_policy)


class AuthoringSearchConfiguration(SearchConfiguration):
    algorithm: SearchAlgorithmConfiguration = Field(
        default_factory=default_search_algorithm
    )
    budget: SearchBudgetConfiguration
    stopping: SearchStoppingConfiguration = FullBudgetStopping(kind="full_budget")
    fpu: FpuConfiguration = Field(default_factory=default_fpu)
    root_exploration: RootExplorationConfiguration = Field(
        default_factory=default_root_exploration
    )
    temperature: TemperatureConfiguration = Field(default_factory=default_temperature)
    tree_reuse: TreeReuseConfiguration = Field(default_factory=default_tree_reuse)
    inference: SearchInferenceConfiguration = Field(
        default_factory=default_search_inference
    )
    backup_discount: float = Field(default=1, gt=0, le=1)


class AuthoringReplayConfiguration(ReplayConfiguration):
    shard_directory: PurePosixPath | None = None


def default_replay() -> AuthoringReplayConfiguration:
    return AuthoringReplayConfiguration(
        capacity_positions=2_500_000,
        shard_directory=None,
        maximum_positions_per_shard=16_384,
        payload_schema_version=1,
        compression="none",
        sampling="uniform",
        credits=default_replay_credits(),
    )


class AuthoringEvaluationConfiguration(FrozenModel):
    search: SearchConfiguration = Field(default_factory=default_evaluation_search)
    paired_games_per_checkpoint: PositiveInt = 200
    bootstrap_samples: PositiveInt = 10_000
    confidence_method: Literal["paired_bootstrap"] = "paired_bootstrap"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    opponent: GoOpponentConfiguration = Field(
        default_factory=default_evaluation_opponent
    )
    komi_half_points: int | None = Field(default=None, strict=True)


class AuthoringGoExperimentConfiguration(FrozenModel):
    schema_version: Literal[2] = 2
    game: Literal["go"] = "go"
    experiment: AuthoringExperimentConfiguration
    search: AuthoringSearchConfiguration
    hardware: HardwareConfiguration = Field(default_factory=planned_hardware_profile)
    topology: TopologyConfiguration = Field(default_factory=default_topology)
    game_configuration: GoGameConfiguration = Field(default_factory=default_game)
    model: ModelConfiguration = Field(default_factory=default_model)
    self_play: SelfPlayConfiguration = Field(default_factory=default_self_play)
    replay: AuthoringReplayConfiguration = Field(default_factory=default_replay)
    training: TrainingConfiguration = Field(default_factory=default_training)
    evaluation: AuthoringEvaluationConfiguration = Field(
        default_factory=AuthoringEvaluationConfiguration
    )
    telemetry: TelemetryConfiguration = Field(default_factory=default_telemetry)
    retention: RetentionConfiguration = Field(default_factory=default_retention)


class AuthoringChessExperimentConfiguration(FrozenModel):
    schema_version: Literal[2] = 2
    game: Literal["chess"] = "chess"
    experiment: AuthoringExperimentConfiguration
    search: AuthoringSearchConfiguration
    hardware: HardwareConfiguration
    topology: TopologyConfiguration
    game_configuration: ChessGameConfiguration
    model: ChessModelConfiguration
    self_play: SelfPlayConfiguration
    replay: ChessReplayConfiguration
    training: ChessTrainingConfiguration
    evaluation: ChessEvaluationConfiguration
    telemetry: TelemetryConfiguration
    retention: RetentionConfiguration


AuthoringRunConfiguration = Annotated[
    AuthoringGoExperimentConfiguration | AuthoringChessExperimentConfiguration,
    Field(discriminator="game"),
]

AUTHORING_RUN_CONFIGURATION_ADAPTER = TypeAdapter(AuthoringRunConfiguration)

AuthoringConfigurationInput = (
    Mapping[str, JsonValue]
    | AuthoringGoExperimentConfiguration
    | AuthoringChessExperimentConfiguration
)


def validate_authoring_configuration(
    value: AuthoringConfigurationInput,
) -> AuthoringRunConfiguration:
    return AUTHORING_RUN_CONFIGURATION_ADAPTER.validate_python(value)


def validate_authoring_configuration_json(
    contents: str | bytes,
) -> AuthoringRunConfiguration:
    return AUTHORING_RUN_CONFIGURATION_ADAPTER.validate_json(contents)


def resolve_configuration(
    authoring: AuthoringRunConfiguration,
) -> ResolvedRunConfiguration:
    match authoring:
        case AuthoringGoExperimentConfiguration():
            return _resolve_go_configuration(authoring)
        case AuthoringChessExperimentConfiguration():
            return _resolve_chess_configuration(authoring)


def _resolve_go_configuration(
    authoring: AuthoringGoExperimentConfiguration,
) -> GoExperimentConfiguration:
    return GoExperimentConfiguration(
        schema_version=2,
        game="go",
        experiment=_resolve_experiment(authoring.experiment),
        hardware=authoring.hardware,
        topology=authoring.topology,
        game_configuration=authoring.game_configuration,
        model=authoring.model,
        search=_resolve_search(authoring.search),
        self_play=authoring.self_play,
        replay=_resolve_replay(authoring),
        training=authoring.training,
        evaluation=_resolve_evaluation(authoring),
        telemetry=authoring.telemetry,
        retention=authoring.retention,
    )


def _resolve_chess_configuration(
    authoring: AuthoringChessExperimentConfiguration,
) -> ChessExperimentConfiguration:
    return ChessExperimentConfiguration(
        schema_version=2,
        game="chess",
        experiment=_resolve_experiment(authoring.experiment),
        hardware=authoring.hardware,
        topology=authoring.topology,
        game_configuration=authoring.game_configuration,
        model=authoring.model,
        search=_resolve_search(authoring.search),
        self_play=authoring.self_play,
        replay=authoring.replay,
        training=authoring.training,
        evaluation=authoring.evaluation,
        telemetry=authoring.telemetry,
        retention=authoring.retention,
    )


def _resolve_experiment(
    authoring: AuthoringExperimentConfiguration,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        name=authoring.name,
        arm_id=authoring.arm_id,
        hypothesis=authoring.hypothesis,
        root_seed=authoring.root_seed,
        duration_seconds=authoring.duration_seconds,
        checkpoint_elapsed_seconds=authoring.checkpoint_elapsed_seconds,
        output_directory=authoring.output_directory,
        manifest_policy=authoring.manifest_policy,
    )


def _resolve_search(authoring: AuthoringSearchConfiguration) -> SearchConfiguration:
    return SearchConfiguration(
        algorithm=authoring.algorithm,
        budget=authoring.budget,
        stopping=authoring.stopping,
        fpu=authoring.fpu,
        root_exploration=authoring.root_exploration,
        temperature=authoring.temperature,
        tree_reuse=authoring.tree_reuse,
        inference=authoring.inference,
        backup_discount=authoring.backup_discount,
    )


def _resolve_replay(
    authoring: AuthoringGoExperimentConfiguration,
) -> ReplayConfiguration:
    shard_directory = (
        authoring.replay.shard_directory
        or authoring.experiment.output_directory / "replay"
    )
    return ReplayConfiguration(
        capacity_positions=authoring.replay.capacity_positions,
        shard_directory=shard_directory,
        maximum_positions_per_shard=authoring.replay.maximum_positions_per_shard,
        payload_schema_version=authoring.replay.payload_schema_version,
        compression=authoring.replay.compression,
        sampling=authoring.replay.sampling,
        credits=authoring.replay.credits,
    )


def _resolve_evaluation(
    authoring: AuthoringGoExperimentConfiguration,
) -> EvaluationConfiguration:
    evaluation = authoring.evaluation
    return EvaluationConfiguration(
        search=evaluation.search,
        checkpoint_elapsed_seconds=authoring.experiment.checkpoint_elapsed_seconds,
        paired_games_per_checkpoint=evaluation.paired_games_per_checkpoint,
        bootstrap_samples=evaluation.bootstrap_samples,
        confidence_method=evaluation.confidence_method,
        confidence_level=evaluation.confidence_level,
        bootstrap_seed=(
            evaluation.bootstrap_seed
            if evaluation.bootstrap_seed is not None
            else authoring.experiment.root_seed
        ),
        suite=GoEvaluationSuite(
            kind="go_paired",
            opponent=evaluation.opponent,
            alternate_colors=True,
            komi_half_points=(
                evaluation.komi_half_points
                if evaluation.komi_half_points is not None
                else authoring.game_configuration.komi_half_points
            ),
        ),
    )
