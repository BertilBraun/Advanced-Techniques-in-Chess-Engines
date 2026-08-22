from __future__ import annotations

from pathlib import Path

import pytest
from src.evaluation.contracts import CandidateOutcome, EvaluationGameResult, EvaluationTerminationReason

pytest.importorskip('AlphaZeroCpp')
from tools.run_stockfish_ladder import LadderProbe, _ladder_elo_fit, _pair_scores, _score_bracket


def _game(game_index: int, pair_index: int, outcome: CandidateOutcome) -> EvaluationGameResult:
    return EvaluationGameResult(
        game_index=game_index,
        pair_index=pair_index,
        opening_id=f'opening-{pair_index}',
        candidate_player='first' if game_index % 2 == 0 else 'second',
        pair_seed=0,
        initial_action_ids=(),
        played_action_ids=(),
        outcome=outcome,
        termination_reason=EvaluationTerminationReason.NATURAL,
        plies=0,
        duration_seconds=1.0,
    )


def _probe(nodes: int, score: float) -> LadderProbe:
    return LadderProbe(
        stockfish_nodes=nodes,
        result_path=Path(f'{nodes}.json'),
        wins=0,
        draws=20,
        losses=0,
        score=score,
        score_confidence_low=max(0.0, score - 0.2),
        score_confidence_high=min(1.0, score + 0.2),
    )


def test_score_bracket_finds_adjacent_node_rungs() -> None:
    bracket = _score_bracket((_probe(5_000, 0.3), _probe(1_000, 0.7), _probe(2_000, 0.55)))

    assert bracket is not None
    assert bracket.lower_stockfish_nodes == 2_000
    assert bracket.upper_stockfish_nodes == 5_000


def test_score_bracket_is_absent_without_crossing() -> None:
    assert _score_bracket((_probe(1_000, 0.8), _probe(2_000, 0.7))) is None


def test_pair_scores_average_color_swapped_games() -> None:
    games = (
        _game(0, 0, CandidateOutcome.WIN),
        _game(1, 0, CandidateOutcome.LOSS),
        _game(2, 1, CandidateOutcome.DRAW),
        _game(3, 1, CandidateOutcome.WIN),
    )

    assert _pair_scores(games) == (0.5, 0.75)


def test_pair_scores_reject_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match='exactly two'):
        _pair_scores((_game(0, 0, CandidateOutcome.WIN),))


def test_ladder_elo_fit_requires_agreed_anchors_for_every_probed_rung() -> None:
    assert _ladder_elo_fit({30: (0.5, 0.5), 55: (0.5, 0.5)}, bootstrap_seed=1) is None


def test_ladder_elo_fit_recovers_a_consistent_rating_with_a_tight_interval() -> None:
    rating = 1500.0
    pair_scores_by_nodes = {
        nodes: (1.0 / (1.0 + 10.0 ** ((anchor - rating) / 400.0)),) * 10
        for nodes, anchor in ((30, 1100.0), (100, 1200.0), (300, 1400.0), (1000, 1700.0))
    }

    fit = _ladder_elo_fit(pair_scores_by_nodes, bootstrap_seed=3)

    assert fit is not None
    assert fit.ladder_elo == pytest.approx(rating, abs=1.0)
    assert fit.confidence_low == pytest.approx(rating, abs=30.0)
    assert fit.confidence_high == pytest.approx(rating, abs=30.0)
    assert fit.confidence_low <= fit.ladder_elo <= fit.confidence_high
