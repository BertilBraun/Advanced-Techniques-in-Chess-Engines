from __future__ import annotations

import random

import numpy as np
from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult, MatchAggregate


def aggregate_match(
    games: tuple[EvaluationGameResult, ...],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> MatchAggregate:
    if not games or len(games) % 2:
        raise ValueError('Match aggregation requires a nonempty even number of games.')
    if bootstrap_samples <= 0:
        raise ValueError('Bootstrap sample count must be positive.')
    pair_scores: list[float] = []
    for game_index in range(0, len(games), 2):
        pair = games[game_index : game_index + 2]
        if pair[0].pair_index != pair[1].pair_index:
            raise ValueError('Adjacent match games must form one player-swapped pair.')
        pair_scores.append(sum(_score(game.outcome) for game in pair) / 2.0)
    generator = random.Random(bootstrap_seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        bootstrap[sample_index] = sum(generator.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
    wins = sum(game.outcome is CandidateOutcome.WIN for game in games)
    draws = sum(game.outcome is CandidateOutcome.DRAW for game in games)
    losses = sum(game.outcome is CandidateOutcome.LOSS for game in games)
    score = (wins + draws * 0.5) / len(games)
    first_player_games = tuple(game for game in games if game.candidate_player == 'first')
    second_player_games = tuple(game for game in games if game.candidate_player == 'second')
    return MatchAggregate(
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        first_player_score=sum(_score(game.outcome) for game in first_player_games) / len(first_player_games),
        second_player_score=sum(_score(game.outcome) for game in second_player_games) / len(second_player_games),
        pair_count=len(pair_scores),
        score_confidence_low=float(np.quantile(bootstrap, 0.025)),
        score_confidence_high=float(np.quantile(bootstrap, 0.975)),
    )


def _score(outcome: CandidateOutcome) -> float:
    match outcome:
        case CandidateOutcome.WIN:
            return 1.0
        case CandidateOutcome.DRAW:
            return 0.5
        case CandidateOutcome.LOSS:
            return 0.0
