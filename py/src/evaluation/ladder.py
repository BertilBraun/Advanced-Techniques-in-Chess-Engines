from __future__ import annotations

import random
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult, EvaluationTerminationReason

# Agreed Melonimarco SSDF-scale anchors for the Stockfish 13 fixed-node rungs
# (chess-recovery-plan-20260820.md, section 0). Extend only with calibrated rungs.
STOCKFISH_FIXED_NODES_ANCHOR_ELO = MappingProxyType(
    {
        30: 1100.0,
        100: 1200.0,
        300: 1400.0,
        1000: 1700.0,
        # Log-interpolated from the same Melonimarco curve: 2000 sits on a plotted point at 1890,
        # 3000 falls between the plotted (2000, 1890) and (5000, 2220) at roughly 2040. The four
        # anchors above reproduce from that curve to within 70 Elo, three of them within 20.
        2000: 1890.0,
        3000: 2040.0,
    }
)

_OUTCOME_SCORE = MappingProxyType(
    {
        CandidateOutcome.WIN: 1.0,
        CandidateOutcome.DRAW: 0.5,
        CandidateOutcome.LOSS: 0.0,
    }
)

_MINIMUM_FIT_ELO = 0.0
_MAXIMUM_FIT_ELO = 5000.0
_BISECTION_ITERATIONS = 80


@dataclass(frozen=True)
class LadderRungObservation:
    anchor_elo: float
    score: float
    game_count: int

    def __post_init__(self) -> None:
        if not isfinite(self.anchor_elo):
            raise ValueError('Ladder anchor Elo must be finite.')
        if not 0.0 <= self.score <= 1.0:
            raise ValueError('Ladder rung score must lie in [0, 1].')
        if self.game_count <= 0:
            raise ValueError('Ladder rung game count must be positive.')


@dataclass(frozen=True)
class LadderRungPairScores:
    anchor_elo: float
    pair_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.anchor_elo):
            raise ValueError('Ladder anchor Elo must be finite.')
        if not self.pair_scores:
            raise ValueError('Ladder rung pair scores must be nonempty.')
        if any(not 0.0 <= score <= 1.0 for score in self.pair_scores):
            raise ValueError('Ladder pair scores must lie in [0, 1].')


def ladder_rung_observation(
    anchor_elo: float,
    games: tuple[EvaluationGameResult, ...],
) -> LadderRungObservation | None:
    # Ply-cap adjudications are draw artifacts, not strength evidence; a rung
    # without any naturally finished game carries no ladder signal.
    rated_games = tuple(
        game for game in games if game.termination_reason is not EvaluationTerminationReason.MAXIMUM_PLIES
    )
    if not rated_games:
        return None
    return LadderRungObservation(
        anchor_elo=anchor_elo,
        score=sum(_OUTCOME_SCORE[game.outcome] for game in rated_games) / len(rated_games),
        game_count=len(rated_games),
    )


def fit_ladder_elo(observations: tuple[LadderRungObservation, ...]) -> float:
    if not observations:
        raise ValueError('Ladder Elo fit requires at least one rung observation.')
    low = _MINIMUM_FIT_ELO
    high = _MAXIMUM_FIT_ELO
    for _ in range(_BISECTION_ITERATIONS):
        rating = (low + high) / 2.0
        residual = 0.0
        for observation in observations:
            expected = 1.0 / (1.0 + 10.0 ** ((observation.anchor_elo - rating) / 400.0))
            residual += observation.game_count * (observation.score - expected)
        if residual > 0.0:
            low = rating
        else:
            high = rating
    return (low + high) / 2.0


def bootstrap_ladder_elo_interval(
    rungs: tuple[LadderRungPairScores, ...],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[float, float]:
    # Resamples rungs as well as opening pairs so the interval carries rung-level
    # sampling, which per-rung game bootstraps understate.
    if not rungs:
        raise ValueError('Ladder bootstrap requires at least one rung.')
    if bootstrap_samples <= 0:
        raise ValueError('Ladder bootstrap sample count must be positive.')
    generator = random.Random(random_seed)
    fits: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_rungs = tuple(rungs[generator.randrange(len(rungs))] for _ in rungs)
        observations = tuple(
            LadderRungObservation(
                anchor_elo=rung.anchor_elo,
                score=(
                    sum(rung.pair_scores[generator.randrange(len(rung.pair_scores))] for _ in rung.pair_scores)
                    / len(rung.pair_scores)
                ),
                game_count=2 * len(rung.pair_scores),
            )
            for rung in sampled_rungs
        )
        fits.append(fit_ladder_elo(observations))
    fits.sort()
    return _quantile(fits, 0.025), _quantile(fits, 0.975)


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
