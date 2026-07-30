from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from src.az.config.base import FrozenModel
from src.az.config.evaluation import EvaluationConfiguration
from src.az.config.experiment import ExperimentConfiguration, HardwareConfiguration
from src.az.config.game import GameConfiguration
from src.az.config.model import ModelConfiguration, ProgressiveModelSchedule
from src.az.config.runtime import RetentionConfiguration, TelemetryConfiguration, TopologyConfiguration
from src.az.config.search import ProgressiveSearchBudget, SearchConfiguration
from src.az.config.training import ReplayConfiguration, SelfPlayConfiguration, TrainingConfiguration
from src.az.games.go.configuration import validate_go_experiment_compatibility


class ResolvedRunConfiguration(FrozenModel):
    schema_version: Literal[2]
    experiment: ExperimentConfiguration
    hardware: HardwareConfiguration
    topology: TopologyConfiguration
    game: GameConfiguration
    model: ModelConfiguration
    search: SearchConfiguration
    self_play: SelfPlayConfiguration
    replay: ReplayConfiguration
    training: TrainingConfiguration
    evaluation: EvaluationConfiguration
    telemetry: TelemetryConfiguration
    retention: RetentionConfiguration

    @model_validator(mode='after')
    def validate_run(self) -> ResolvedRunConfiguration:
        duration = self.experiment.duration_seconds
        match self.search.budget:
            case ProgressiveSearchBudget(stages=stages) if any(
                stage.start_elapsed_seconds >= duration for stage in stages
            ):
                raise ValueError('Every progressive search stage must start before the run ends.')
            case _:
                pass
        match self.model.schedule:
            case ProgressiveModelSchedule(stages=stages) if any(
                stage.start_elapsed_seconds >= duration for stage in stages
            ):
                raise ValueError('Every progressive model stage must start before the run ends.')
            case _:
                pass
        if self.training.global_batch_size != (self.training.local_batch_size * len(self.topology.trainer.device_ids)):
            raise ValueError('Global batch size must equal local batch size times trainer ranks.')
        configured_devices = (
            self.topology.trainer.device_ids + self.topology.self_play.device_ids + self.topology.evaluation.device_ids
        )
        if max(configured_devices) >= self.hardware.expected_gpu_count:
            raise ValueError('Topology device IDs must be below the expected GPU count.')
        if self.evaluation.checkpoint_elapsed_seconds != self.experiment.checkpoint_elapsed_seconds:
            raise ValueError('Evaluation and experiment checkpoint schedules must match.')
        if self.evaluation.suite.komi_half_points != self.game.komi_half_points:
            raise ValueError('Evaluation and training komi must match.')
        validate_go_experiment_compatibility(
            self.training.objective,
            self.training.optimizer.weight_decay,
            self.replay.payload_schema_version,
        )
        return self
