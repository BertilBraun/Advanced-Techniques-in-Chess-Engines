from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class RandomOpeningStartParameters:
    kind: Literal['random_opening']
    maximum_plies: int

    def __post_init__(self) -> None:
        if self.maximum_plies < 0:
            raise ValueError('Maximum random opening plies must be nonnegative.')


@dataclass(frozen=True)
class RestartStateStartParameters:
    kind: Literal['restart_state']
    true_start_probability: float
    candidate_visit_mass: float
    minimum_candidates: int
    maximum_candidates: int
    maximum_absolute_root_value: float
    minimum_remaining_plies: int
    maximum_archive_positions: int
    maximum_age_generations: int

    def __post_init__(self) -> None:
        if not 0.0 < self.true_start_probability <= 1.0:
            raise ValueError('True-start probability must lie in (0, 1].')
        if not 0.0 < self.candidate_visit_mass <= 1.0:
            raise ValueError('Candidate visit mass must lie in (0, 1].')
        if self.minimum_candidates < 2 or self.maximum_candidates < self.minimum_candidates:
            raise ValueError('Restart candidate bounds are invalid.')
        if not 0.0 <= self.maximum_absolute_root_value <= 1.0:
            raise ValueError('Maximum absolute restart root value must lie in [0, 1].')
        if self.minimum_remaining_plies <= 0:
            raise ValueError('Minimum remaining restart plies must be positive.')
        if self.maximum_archive_positions <= 0 or self.maximum_age_generations <= 0:
            raise ValueError('Restart archive capacity and maximum age must be positive.')


StartPositionParameters: TypeAlias = RandomOpeningStartParameters | RestartStateStartParameters


@dataclass(frozen=True)
class ZeroFirstPlayUrgencyParameters:
    kind: Literal['zero'] = 'zero'


@dataclass(frozen=True)
class ParentValueFirstPlayUrgencyParameters:
    kind: Literal['parent_value'] = 'parent_value'


@dataclass(frozen=True)
class ReducedParentValueFirstPlayUrgencyParameters:
    reduction: float
    kind: Literal['reduced_parent_value'] = 'reduced_parent_value'

    def __post_init__(self) -> None:
        if not isfinite(self.reduction) or self.reduction <= 0.0:
            raise ValueError('Reduced-parent FPU reduction must be finite and positive.')


FirstPlayUrgencyParameters: TypeAlias = (
    ZeroFirstPlayUrgencyParameters
    | ParentValueFirstPlayUrgencyParameters
    | ReducedParentValueFirstPlayUrgencyParameters
)


@dataclass(frozen=True)
class FixedFullSearchBudget:
    kind: Literal['fixed']
    visits: int

    def __post_init__(self) -> None:
        if self.visits <= 0:
            raise ValueError('Fixed full-search visits must be positive.')


@dataclass(frozen=True)
class AdaptiveFullSearchBudget:
    kind: Literal['adaptive']
    minimum_visits: int
    maximum_visits: int
    observation_interval: int
    leader_stability_window: int
    root_value_tolerance: float
    initial_top_visit_share: float
    final_top_visit_share: float
    initial_top_two_margin: float
    final_top_two_margin: float
    threshold_relaxation_visits: int
    minimum_search_correction_to_unlock_tail: float | None

    def __post_init__(self) -> None:
        if self.minimum_visits <= 0 or self.maximum_visits < self.minimum_visits:
            raise ValueError('Adaptive full-search visit bounds are invalid.')
        if (
            self.observation_interval <= 0
            or self.leader_stability_window < self.observation_interval
            or self.threshold_relaxation_visits <= 0
        ):
            raise ValueError('Adaptive observation cadence is invalid.')
        if self.leader_stability_window % self.observation_interval:
            raise ValueError('Adaptive leader window must divide into observation intervals.')
        thresholds = (
            self.root_value_tolerance,
            self.initial_top_visit_share,
            self.final_top_visit_share,
            self.initial_top_two_margin,
            self.final_top_two_margin,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError('Adaptive search thresholds must be finite probabilities.')
        if self.minimum_search_correction_to_unlock_tail is not None and not (
            0.0 <= self.minimum_search_correction_to_unlock_tail <= 1.0
        ):
            raise ValueError('Search-correction threshold must lie in [0, 1].')


FullSearchBudget: TypeAlias = FixedFullSearchBudget | AdaptiveFullSearchBudget


@dataclass(frozen=True)
class ResolvedSelfPlayParameters:
    start_position: StartPositionParameters
    full_search_probability: float
    parallel_searches: int
    full_search_budget: FullSearchBudget
    fast_searches: int
    forced_playout_coefficient: float
    exploration_constant: float
    first_play_urgency: FirstPlayUrgencyParameters
    dirichlet_alpha: float
    dirichlet_epsilon: float
    retained_root_visit_fraction: float
    starting_temperature: float
    final_temperature: float
    greedy_after_ply: int
    maximum_game_plies: int | None
    primary_sample_weight: float
    value_discount_per_ply: float
    force_fast_search_after_ply: int | None = None
    virtual_loss_weight: float = 1.0
    bootstrap_cut_game_value: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.full_search_probability <= 1.0:
            raise ValueError('Full-search probability must lie in (0, 1].')
        if self.parallel_searches <= 0:
            raise ValueError('Parallel searches must be positive.')
        if self.maximum_full_searches <= self.parallel_searches:
            raise ValueError('Full-search budget must exceed parallel searches.')
        if self.fast_searches <= 0:
            raise ValueError('Fast-search budget must be positive.')
        if not isfinite(self.forced_playout_coefficient) or self.forced_playout_coefficient < 0.0:
            raise ValueError('Forced-playout coefficient must be finite and nonnegative.')
        if self.exploration_constant <= 0.0 or self.dirichlet_alpha <= 0.0:
            raise ValueError('Search constants must be positive.')
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError('Dirichlet epsilon must lie in [0, 1].')
        if not 0.0 <= self.retained_root_visit_fraction <= 1.0:
            raise ValueError('Retained-root fraction must lie in [0, 1].')
        if self.starting_temperature <= 0.0 or self.final_temperature <= 0.0:
            raise ValueError('Temperatures must be positive.')
        if self.final_temperature > self.starting_temperature:
            raise ValueError('Final self-play temperature cannot exceed the starting temperature.')
        if self.greedy_after_ply <= 0:
            raise ValueError('Greedy ply must be positive.')
        if self.force_fast_search_after_ply is not None and self.force_fast_search_after_ply <= 0:
            raise ValueError('Forced-fast-search ply must be positive.')
        if not isfinite(self.virtual_loss_weight) or not 0.0 <= self.virtual_loss_weight <= 1.0:
            raise ValueError('Virtual-loss weight must be finite and lie in [0, 1].')
        match self.start_position:
            case RandomOpeningStartParameters(maximum_plies=maximum_plies):
                if self.maximum_game_plies is not None and self.maximum_game_plies <= maximum_plies:
                    raise ValueError('Maximum game plies must exceed maximum random opening plies.')
            case RestartStateStartParameters():
                pass
        if self.primary_sample_weight <= 0.0:
            raise ValueError('Primary sample weight must be positive.')
        if not isfinite(self.value_discount_per_ply) or not 0.0 < self.value_discount_per_ply <= 1.0:
            raise ValueError('Value discount per ply must be finite and lie in (0, 1].')

    @property
    def maximum_full_searches(self) -> int:
        match self.full_search_budget:
            case FixedFullSearchBudget(visits=visits):
                return visits
            case AdaptiveFullSearchBudget(maximum_visits=maximum_visits):
                return maximum_visits
