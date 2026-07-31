from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.evaluation import EvaluationConfiguration
from src.az.config.experiment import ExperimentConfiguration, HardwareConfiguration
from src.az.config.model import ModelConfiguration
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.runtime import (
    RetentionConfiguration,
    TelemetryConfiguration,
    TopologyConfiguration,
)
from src.az.config.search import (
    FpuConfiguration,
    RootExplorationConfiguration,
    SearchAlgorithmConfiguration,
    SearchBudgetConfiguration,
    SearchConfiguration,
    SearchStoppingConfiguration,
    TemperatureConfiguration,
    TreeReuseConfiguration,
    SearchInferenceConfiguration,
)
from src.az.config.serialization import model_sha256
from src.az.config.training import (
    ReplayConfiguration,
    SelfPlayConfiguration,
    TrainingConfiguration,
)
from src.az.games.go.configuration import GoGameConfiguration


class SearchComputeArmDefinition(FrozenModel):
    arm_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    output_directory: PurePosixPath
    budget: SearchBudgetConfiguration
    stopping: SearchStoppingConfiguration


class FpuArmDefinition(FrozenModel):
    arm_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    output_directory: PurePosixPath
    fpu: FpuConfiguration


class CommonControls(FrozenModel):
    hardware: HardwareConfiguration
    topology: TopologyConfiguration
    game: GoGameConfiguration
    model: ModelConfiguration
    self_play: SelfPlayConfiguration
    replay: ReplayConfiguration
    training: TrainingConfiguration
    evaluation: EvaluationConfiguration
    telemetry: TelemetryConfiguration
    retention: RetentionConfiguration
    duration_seconds: int = Field(gt=0)
    checkpoint_elapsed_seconds: tuple[int, ...]
    search_algorithm: SearchAlgorithmConfiguration
    search_root_exploration: RootExplorationConfiguration
    search_temperature: TemperatureConfiguration
    search_tree_reuse: TreeReuseConfiguration
    search_inference: SearchInferenceConfiguration
    search_backup_discount: float
    search_budget: SearchBudgetConfiguration | None
    search_stopping: SearchStoppingConfiguration | None
    search_fpu: FpuConfiguration | None


class SearchComputeAblationMatrix(FrozenModel):
    kind: Literal["search_compute"]
    schema_version: Literal[1] = 1
    matrix_id: UUID
    common_configuration: ResolvedRunConfiguration
    arms: tuple[SearchComputeArmDefinition, ...] = Field(min_length=2)
    root_seeds: tuple[int, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_matrix(self) -> SearchComputeAblationMatrix:
        _validate_identities(self.arms, self.root_seeds)
        return self


class FpuAblationMatrix(FrozenModel):
    kind: Literal["fpu"]
    schema_version: Literal[1] = 1
    matrix_id: UUID
    common_configuration: ResolvedRunConfiguration
    arms: tuple[FpuArmDefinition, ...] = Field(min_length=2)
    root_seeds: tuple[int, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_matrix(self) -> FpuAblationMatrix:
        _validate_identities(self.arms, self.root_seeds)
        return self


AblationMatrix = SearchComputeAblationMatrix | FpuAblationMatrix


class ExpandedAblationArm(FrozenModel):
    arm_id: UUID
    seed: int
    factor_identity: str = Field(min_length=1)
    configuration: ResolvedRunConfiguration
    configuration_sha256: Sha256
    common_controls_sha256: Sha256


def _validate_identities(
    arms: tuple[SearchComputeArmDefinition, ...] | tuple[FpuArmDefinition, ...],
    seeds: tuple[int, ...],
) -> None:
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("Ablation arm identities must be unique.")
    if len({arm.output_directory for arm in arms}) != len(arms):
        raise ValueError("Ablation arm output directories must be unique.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ablation root seeds must be unique.")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("Ablation root seeds must be between zero and 2^63 - 1.")


def _controls(
    configuration: ResolvedRunConfiguration,
    matrix_kind: Literal["search_compute", "fpu"],
) -> CommonControls:
    search = configuration.search
    return CommonControls(
        hardware=configuration.hardware,
        topology=configuration.topology,
        game=configuration.game_configuration,
        model=configuration.model,
        self_play=configuration.self_play,
        replay=configuration.replay,
        training=configuration.training,
        evaluation=configuration.evaluation,
        telemetry=configuration.telemetry,
        retention=configuration.retention,
        duration_seconds=configuration.experiment.duration_seconds,
        checkpoint_elapsed_seconds=configuration.experiment.checkpoint_elapsed_seconds,
        search_algorithm=search.algorithm,
        search_root_exploration=search.root_exploration,
        search_temperature=search.temperature,
        search_tree_reuse=search.tree_reuse,
        search_inference=search.inference,
        search_backup_discount=search.backup_discount,
        search_budget=search.budget if matrix_kind == "fpu" else None,
        search_stopping=search.stopping if matrix_kind == "fpu" else None,
        search_fpu=search.fpu if matrix_kind == "search_compute" else None,
    )


def _search(
    base: SearchConfiguration,
    budget: SearchBudgetConfiguration,
    stopping: SearchStoppingConfiguration,
    fpu: FpuConfiguration,
) -> SearchConfiguration:
    return SearchConfiguration(
        algorithm=base.algorithm,
        budget=budget,
        stopping=stopping,
        fpu=fpu,
        root_exploration=base.root_exploration,
        temperature=base.temperature,
        tree_reuse=base.tree_reuse,
        inference=base.inference,
        backup_discount=base.backup_discount,
    )


def _experiment(
    base: ExperimentConfiguration,
    arm_id: str,
    name: str,
    hypothesis: str,
    seed: int,
    output_directory: PurePosixPath,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        name=name,
        arm_id=arm_id,
        hypothesis=hypothesis,
        root_seed=seed,
        duration_seconds=base.duration_seconds,
        checkpoint_elapsed_seconds=base.checkpoint_elapsed_seconds,
        output_directory=output_directory / f"seed-{seed}",
        manifest_policy=base.manifest_policy,
    )


def _configuration(
    base: ResolvedRunConfiguration,
    experiment: ExperimentConfiguration,
    search: SearchConfiguration,
) -> ResolvedRunConfiguration:
    return base.model_copy(update={"experiment": experiment, "search": search})


def expand_matrix(matrix: AblationMatrix) -> tuple[ExpandedAblationArm, ...]:
    base = matrix.common_configuration
    controls_sha256 = model_sha256(_controls(base, matrix.kind))
    expanded: list[ExpandedAblationArm] = []
    for arm in matrix.arms:
        for seed in matrix.root_seeds:
            experiment = _experiment(
                base.experiment,
                arm.arm_id,
                arm.name,
                arm.hypothesis,
                seed,
                arm.output_directory,
            )
            match matrix, arm:
                case SearchComputeAblationMatrix(), SearchComputeArmDefinition():
                    search = _search(
                        base.search, arm.budget, arm.stopping, base.search.fpu
                    )
                case FpuAblationMatrix(), FpuArmDefinition():
                    search = _search(
                        base.search, base.search.budget, base.search.stopping, arm.fpu
                    )
                case _:
                    raise AssertionError("Matrix type fixes its arm definition type.")
            configured = _configuration(base, experiment, search)
            if model_sha256(_controls(configured, matrix.kind)) != controls_sha256:
                raise AssertionError(
                    "Expanded arm changed an undeclared common control."
                )
            identity = f"{arm.arm_id}:seed:{seed}"
            expanded.append(
                ExpandedAblationArm(
                    arm_id=uuid5(matrix.matrix_id, identity),
                    seed=seed,
                    factor_identity=arm.arm_id,
                    configuration=configured,
                    configuration_sha256=model_sha256(configured),
                    common_controls_sha256=controls_sha256,
                )
            )
    return tuple(expanded)
