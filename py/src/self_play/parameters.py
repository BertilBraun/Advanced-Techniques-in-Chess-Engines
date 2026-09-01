from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

from src.search_stopping.policy import SearchStopPolicy


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
class ResolvedSelfPlayParameters:
    start_position: StartPositionParameters
    baseline_visits: int
    search_stop_policy: SearchStopPolicy
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
    virtual_loss_weight: float = 1.0
    bootstrap_cut_game_value: bool = False

    def __post_init__(self) -> None:
        if self.baseline_visits <= 0:
            raise ValueError('Baseline visits must be positive.')
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
