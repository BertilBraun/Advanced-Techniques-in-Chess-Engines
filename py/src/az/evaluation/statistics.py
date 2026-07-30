from __future__ import annotations

import math
import random
from enum import Enum
from typing import Literal

from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.evaluation.models import EvaluationPairResult


class EloBoundaryConvention(str, Enum):
    HALF_GAME_CONTINUITY = 'half_game_continuity'


class ConfidenceInterval(FrozenModel):
    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)


class MatchStatistics(FrozenModel):
    games: int = Field(gt=0)
    pairs: int = Field(gt=0)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    mean_score: float = Field(ge=0, le=1)
    score_confidence_interval: ConfidenceInterval
    elo: float
    elo_confidence_interval: ConfidenceInterval
    elo_boundary_convention: Literal[EloBoundaryConvention.HALF_GAME_CONTINUITY]
    bootstrap_samples: int = Field(gt=0)
    bootstrap_seed: int = Field(ge=0, le=2**63 - 1)
    confidence_method: Literal['paired_bootstrap']


class LearningCurvePoint(FrozenModel):
    elapsed_hours: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    elo: float


class LearningCurveStatistics(FrozenModel):
    observed_start_hours: float = Field(gt=0)
    observed_end_hours: float = Field(gt=0)
    score_auc_score_hours: float = Field(ge=0)
    elo_auc_elo_hours: float
    final_score: float = Field(ge=0, le=1)
    final_elo: float
    final_score_per_hour: float = Field(ge=0)
    final_elo_per_hour: float


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_score_interval(
    pairs: tuple[EvaluationPairResult, ...],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> ConfidenceInterval:
    if not pairs or bootstrap_samples <= 0 or not 0 < confidence_level < 1:
        raise ValueError('Paired bootstrap requires pairs, positive samples, and confidence in (0, 1).')
    pair_scores = tuple(sum(game.candidate_score for game in pair.games) / 2 for pair in pairs)
    random_source = random.Random(seed)
    estimates = tuple(
        sum(pair_scores[random_source.randrange(len(pair_scores))] for _ in pair_scores) / len(pair_scores)
        for _ in range(bootstrap_samples)
    )
    tail = (1 - confidence_level) / 2
    return ConfidenceInterval(
        lower=_quantile(estimates, tail),
        upper=_quantile(estimates, 1 - tail),
        confidence_level=confidence_level,
    )


def score_to_elo(score: float, games: int) -> float:
    if not 0 <= score <= 1 or games <= 0:
        raise ValueError('Elo conversion requires a score in [0, 1] and positive game count.')
    adjusted = 1 / (2 * games) if score == 0 else 1 - 1 / (2 * games) if score == 1 else score
    return 400 * math.log10(adjusted / (1 - adjusted))


def summarize_match(
    pairs: tuple[EvaluationPairResult, ...],
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> MatchStatistics:
    identities = tuple((pair.evaluation_id, pair.pair_index) for pair in pairs)
    if len(set(identities)) != len(identities):
        raise ValueError('Match statistics reject duplicate evaluation pair identities.')
    comparison_identities = {
        (
            pair.evaluation_id,
            pair.games[0].candidate,
            pair.games[0].opponent,
            pair.games[0].requested_elapsed_seconds,
            pair.games[0].published_checkpoint_elapsed_seconds,
            pair.games[0].board_size,
            pair.games[0].komi_half_points,
            pair.games[0].scoring_rule,
            pair.games[0].ko_rule,
            pair.games[0].suicide_rule,
        )
        for pair in pairs
    }
    if len(comparison_identities) != 1:
        raise ValueError('Match statistics require one homogeneous evaluation comparison.')
    games = tuple(game for pair in pairs for game in pair.games)
    if not games:
        raise ValueError('Match statistics require paired games.')
    wins = sum(game.candidate_score == 1 for game in games)
    draws = sum(game.candidate_score == 0.5 for game in games)
    losses = sum(game.candidate_score == 0 for game in games)
    score = sum(game.candidate_score for game in games) / len(games)
    score_interval = paired_bootstrap_score_interval(
        pairs,
        bootstrap_samples,
        confidence_level,
        bootstrap_seed,
    )
    return MatchStatistics(
        games=len(games),
        pairs=len(pairs),
        wins=wins,
        draws=draws,
        losses=losses,
        mean_score=score,
        score_confidence_interval=score_interval,
        elo_confidence_interval=ConfidenceInterval(
            lower=score_to_elo(score_interval.lower, len(games)),
            upper=score_to_elo(score_interval.upper, len(games)),
            confidence_level=confidence_level,
        ),
        elo=score_to_elo(score, len(games)),
        elo_boundary_convention=EloBoundaryConvention.HALF_GAME_CONTINUITY,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_method='paired_bootstrap',
    )


def learning_curve_statistics(points: tuple[LearningCurvePoint, ...]) -> LearningCurveStatistics:
    if not points or tuple(sorted({point.elapsed_hours for point in points})) != tuple(
        point.elapsed_hours for point in points
    ):
        raise ValueError('Learning curve points must have unique increasing elapsed hours.')
    score_auc = sum(
        (right.elapsed_hours - left.elapsed_hours) * (left.score + right.score) / 2
        for left, right in zip(points, points[1:])
    )
    elo_auc = sum(
        (right.elapsed_hours - left.elapsed_hours) * (left.elo + right.elo) / 2
        for left, right in zip(points, points[1:])
    )
    final = points[-1]
    return LearningCurveStatistics(
        observed_start_hours=points[0].elapsed_hours,
        observed_end_hours=final.elapsed_hours,
        score_auc_score_hours=score_auc,
        elo_auc_elo_hours=elo_auc,
        final_score=final.score,
        final_elo=final.elo,
        final_score_per_hour=final.score / final.elapsed_hours,
        final_elo_per_hour=final.elo / final.elapsed_hours,
    )
