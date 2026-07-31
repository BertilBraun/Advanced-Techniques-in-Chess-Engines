from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from src.az.config.base import FrozenModel
from src.az.config.evaluation import EvaluationConfiguration
from src.az.config.experiment import ExperimentConfiguration, HardwareConfiguration
from src.az.config.model import ModelConfiguration, ProgressiveModelSchedule
from src.az.config.runtime import (
    RetentionConfiguration,
    TelemetryConfiguration,
    TopologyConfiguration,
)
from src.az.config.search import ProgressiveSearchBudget, SearchConfiguration
from src.az.config.training import (
    ReplayConfiguration,
    SelfPlayConfiguration,
    TrainingConfiguration,
)
from src.az.games.chess.configuration import (
    ChessEvaluationConfiguration,
    ChessGameConfiguration,
    ChessModelConfiguration,
    ChessReplayConfiguration,
    ChessTrainingConfiguration,
    ProgressiveChessModelSchedule,
    validate_chess_experiment_compatibility,
)
from src.az.games.go.configuration import (
    GoGameConfiguration,
    validate_go_experiment_compatibility,
)


class ExperimentConfigurationBase(FrozenModel):
    schema_version: Literal[2]
    experiment: ExperimentConfiguration
    hardware: HardwareConfiguration
    topology: TopologyConfiguration
    search: SearchConfiguration
    self_play: SelfPlayConfiguration
    telemetry: TelemetryConfiguration
    retention: RetentionConfiguration

    @model_validator(mode="after")
    def validate_shared_configuration(self) -> ExperimentConfigurationBase:
        duration = self.experiment.duration_seconds
        match self.search.budget:
            case ProgressiveSearchBudget(stages=stages) if any(
                stage.start_elapsed_seconds >= duration for stage in stages
            ):
                raise ValueError(
                    "Every progressive search stage must start before the run ends."
                )
            case _:
                pass
        configured_devices = (
            self.topology.trainer.device_ids
            + self.topology.self_play.device_ids
            + self.topology.evaluation.device_ids
        )
        if (
            self.hardware.expected_gpu_count > 0
            and max(configured_devices) >= self.hardware.expected_gpu_count
        ):
            raise ValueError(
                "Topology device IDs must be below the expected GPU count."
            )
        if self.telemetry.search_trace_checkpoints and (
            self.telemetry.search_trace_checkpoints[-1]
            >= self.search.minimum_budget_cap
        ):
            raise ValueError(
                "Every search trace checkpoint must be below every applicable search cap."
            )
        return self


class GoExperimentConfiguration(ExperimentConfigurationBase):
    game: Literal["go"]
    game_configuration: GoGameConfiguration
    model: ModelConfiguration
    replay: ReplayConfiguration
    training: TrainingConfiguration
    evaluation: EvaluationConfiguration

    @model_validator(mode="after")
    def validate_go_configuration(self) -> GoExperimentConfiguration:
        if (
            self.evaluation.checkpoint_elapsed_seconds
            != self.experiment.checkpoint_elapsed_seconds
        ):
            raise ValueError(
                "Evaluation and experiment checkpoint schedules must match."
            )
        match self.model.schedule:
            case ProgressiveModelSchedule(stages=stages) if any(
                stage.start_elapsed_seconds >= self.experiment.duration_seconds
                for stage in stages
            ):
                raise ValueError(
                    "Every progressive model stage must start before the run ends."
                )
            case _:
                pass
        if self.training.global_batch_size != (
            self.training.local_batch_size * len(self.topology.trainer.device_ids)
        ):
            raise ValueError(
                "Global batch size must equal local batch size times trainer ranks."
            )
        if (
            self.evaluation.suite.komi_half_points
            != self.game_configuration.komi_half_points
        ):
            raise ValueError("Evaluation and training komi must match.")
        validate_go_experiment_compatibility(
            self.training.objective,
            self.training.optimizer.weight_decay,
            self.replay.payload_schema_version,
        )
        return self


class ChessExperimentConfiguration(ExperimentConfigurationBase):
    game: Literal["chess"]
    game_configuration: ChessGameConfiguration
    model: ChessModelConfiguration
    replay: ChessReplayConfiguration
    training: ChessTrainingConfiguration
    evaluation: ChessEvaluationConfiguration

    @model_validator(mode="after")
    def validate_chess_configuration(self) -> ChessExperimentConfiguration:
        if (
            self.evaluation.checkpoint_elapsed_seconds
            != self.experiment.checkpoint_elapsed_seconds
        ):
            raise ValueError(
                "Evaluation and experiment checkpoint schedules must match."
            )
        match self.model.schedule:
            case ProgressiveChessModelSchedule(stages=stages) if any(
                stage.start_elapsed_seconds >= self.experiment.duration_seconds
                for stage in stages
            ):
                raise ValueError(
                    "Every progressive model stage must start before the run ends."
                )
            case _:
                pass
        if self.training.global_batch_size != (
            self.training.local_batch_size * len(self.topology.trainer.device_ids)
        ):
            raise ValueError(
                "Global batch size must equal local batch size times trainer ranks."
            )
        validate_chess_experiment_compatibility(
            self.training.objective,
            self.training.optimizer.weight_decay,
        )
        return self


ResolvedRunConfiguration = Annotated[
    GoExperimentConfiguration | ChessExperimentConfiguration,
    Field(discriminator="game"),
]

RESOLVED_RUN_CONFIGURATION_ADAPTER = TypeAdapter(ResolvedRunConfiguration)

ResolvedConfigurationInput = (
    Mapping[str, JsonValue] | GoExperimentConfiguration | ChessExperimentConfiguration
)


def validate_resolved_configuration(
    value: ResolvedConfigurationInput,
) -> ResolvedRunConfiguration:
    return RESOLVED_RUN_CONFIGURATION_ADAPTER.validate_python(value)


def validate_resolved_configuration_json(
    contents: str | bytes,
) -> ResolvedRunConfiguration:
    return RESOLVED_RUN_CONFIGURATION_ADAPTER.validate_json(contents)
