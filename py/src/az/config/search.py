from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.az.config.base import FrozenModel


class PuctSearchConfiguration(FrozenModel):
    kind: Literal['puct']
    exploration_constant: PositiveFloat


SearchAlgorithmConfiguration = PuctSearchConfiguration


class FixedSearchBudget(FrozenModel):
    kind: Literal['fixed']
    simulations: PositiveInt


class SearchBudgetStage(FrozenModel):
    start_elapsed_seconds: int = Field(ge=0)
    simulations: PositiveInt


class ProgressiveSearchBudget(FrozenModel):
    kind: Literal['progressive']
    stages: tuple[SearchBudgetStage, ...] = Field(min_length=2)

    @model_validator(mode='after')
    def validate_stages(self) -> ProgressiveSearchBudget:
        starts = tuple(stage.start_elapsed_seconds for stage in self.stages)
        if starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError('Progressive search stages must start at zero and increase strictly.')
        return self


class MixedSearchBudget(FrozenModel):
    kind: Literal['mixed']
    cheap_simulations: PositiveInt
    full_simulations: PositiveInt
    full_search_probability: float = Field(gt=0, lt=1)
    cheap_policy_target_weight: float = Field(ge=0)
    full_policy_target_weight: PositiveFloat

    @model_validator(mode='after')
    def validate_caps(self) -> MixedSearchBudget:
        if self.cheap_simulations >= self.full_simulations:
            raise ValueError('Mixed search requires a cheap cap below the full cap.')
        return self


SearchBudgetConfiguration = Annotated[
    FixedSearchBudget | ProgressiveSearchBudget | MixedSearchBudget,
    Field(discriminator='kind'),
]


class FullBudgetStopping(FrozenModel):
    kind: Literal['full_budget']


class VisitMarginAdaptiveRule(FrozenModel):
    kind: Literal['visit_margin']
    minimum_simulations: PositiveInt
    check_interval_simulations: PositiveInt
    required_top_visit_fraction: float = Field(gt=0.5, le=1)
    required_top_two_margin: float = Field(ge=0, le=1)
    calibration_id: str = Field(min_length=1)


AdaptiveSearchStopping = VisitMarginAdaptiveRule
SearchStoppingConfiguration = Annotated[
    FullBudgetStopping | VisitMarginAdaptiveRule,
    Field(discriminator='kind'),
]


class ParentValueFpu(FrozenModel):
    kind: Literal['parent_value']


class ReducedParentValueFpu(FrozenModel):
    kind: Literal['reduced_parent_value']
    reduction: float = Field(ge=0)


class VisitedChildMeanFpu(FrozenModel):
    kind: Literal['visited_child_mean']
    no_visited_child_value: float = Field(ge=-1, le=1)


FpuConfiguration = Annotated[
    ParentValueFpu | ReducedParentValueFpu | VisitedChildMeanFpu,
    Field(discriminator='kind'),
]


class DisabledRootExploration(FrozenModel):
    kind: Literal['disabled']


class DirichletRootExploration(FrozenModel):
    kind: Literal['dirichlet']
    alpha: PositiveFloat
    exploration_fraction: float = Field(gt=0, lt=1)


RootExplorationConfiguration = Annotated[
    DisabledRootExploration | DirichletRootExploration,
    Field(discriminator='kind'),
]


class ConstantTemperature(FrozenModel):
    kind: Literal['constant']
    temperature: float = Field(ge=0)


class TemperatureStage(FrozenModel):
    maximum_ply_exclusive: PositiveInt
    temperature: float = Field(ge=0)


class PlyTemperatureSchedule(FrozenModel):
    kind: Literal['by_ply']
    stages: tuple[TemperatureStage, ...] = Field(min_length=1)
    final_temperature: float = Field(ge=0)

    @model_validator(mode='after')
    def validate_stages(self) -> PlyTemperatureSchedule:
        limits = tuple(stage.maximum_ply_exclusive for stage in self.stages)
        if tuple(sorted(set(limits))) != limits:
            raise ValueError('Temperature ply limits must increase strictly.')
        return self


TemperatureConfiguration = Annotated[
    ConstantTemperature | PlyTemperatureSchedule,
    Field(discriminator='kind'),
]


class DisabledTreeReuse(FrozenModel):
    kind: Literal['disabled']


class RetainSubtree(FrozenModel):
    kind: Literal['retain_subtree']
    maximum_retained_nodes: PositiveInt


TreeReuseConfiguration = Annotated[DisabledTreeReuse | RetainSubtree, Field(discriminator='kind')]


class SearchInferenceConfiguration(FrozenModel):
    maximum_batch_size: PositiveInt
    maximum_wait_microseconds: int = Field(ge=0)
    cache_capacity: int = Field(ge=0)


class SearchConfiguration(FrozenModel):
    algorithm: SearchAlgorithmConfiguration
    budget: SearchBudgetConfiguration
    stopping: SearchStoppingConfiguration
    fpu: FpuConfiguration
    root_exploration: RootExplorationConfiguration
    temperature: TemperatureConfiguration
    tree_reuse: TreeReuseConfiguration
    inference: SearchInferenceConfiguration
    backup_discount: float = Field(gt=0, le=1)

    @property
    def minimum_budget_cap(self) -> int:
        match self.budget:
            case FixedSearchBudget(simulations=simulations):
                return simulations
            case ProgressiveSearchBudget(stages=stages):
                return min(stage.simulations for stage in stages)
            case MixedSearchBudget(cheap_simulations=cheap_simulations):
                return cheap_simulations

    @model_validator(mode='after')
    def validate_stopping(self) -> SearchConfiguration:
        match self.stopping, self.budget:
            case VisitMarginAdaptiveRule(), MixedSearchBudget():
                raise ValueError('Adaptive stopping cannot be combined with mixed search.')
            case (
                VisitMarginAdaptiveRule(
                    minimum_simulations=minimum_simulations,
                    check_interval_simulations=check_interval,
                ),
                _,
            ):
                if minimum_simulations >= self.minimum_budget_cap:
                    raise ValueError('Adaptive stopping minimum must be below every applicable budget cap.')
                if check_interval > self.minimum_budget_cap:
                    raise ValueError('Adaptive stopping check interval cannot exceed any applicable budget cap.')
            case FullBudgetStopping(), _:
                pass
        return self
