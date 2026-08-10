from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ResolvedSelfPlayParameters:
    maximum_random_opening_plies: int
    full_search_probability: float
    parallel_searches: int
    full_searches: int
    fast_searches: int
    minimum_root_visits: int
    exploration_constant: float
    fpu_reduction: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    retained_root_visit_fraction: float
    starting_temperature: float
    final_temperature: float
    greedy_after_ply: int
    maximum_game_plies: int | None
    primary_sample_weight: float

    def __post_init__(self) -> None:
        if self.maximum_random_opening_plies < 0:
            raise ValueError('Maximum random opening plies must be nonnegative.')
        if not 0.0 < self.full_search_probability <= 1.0:
            raise ValueError('Full-search probability must lie in (0, 1].')
        if self.parallel_searches <= 0:
            raise ValueError('Parallel searches must be positive.')
        if self.full_searches <= self.parallel_searches:
            raise ValueError('Full-search budget must exceed parallel searches.')
        if self.fast_searches <= 0:
            raise ValueError('Fast-search budget must be positive.')
        if self.minimum_root_visits < 0:
            raise ValueError('Minimum root visits must be nonnegative.')
        if self.exploration_constant <= 0.0 or self.dirichlet_alpha <= 0.0:
            raise ValueError('Search constants must be positive.')
        if not isfinite(self.fpu_reduction) or self.fpu_reduction < 0.0:
            raise ValueError('FPU reduction must be finite and nonnegative.')
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
        if self.maximum_game_plies is not None and self.maximum_game_plies <= self.maximum_random_opening_plies:
            raise ValueError('Maximum game plies must exceed maximum random opening plies.')
        if self.primary_sample_weight <= 0.0:
            raise ValueError('Primary sample weight must be positive.')
