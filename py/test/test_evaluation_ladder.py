from __future__ import annotations

import pytest

from src.evaluation.ladder import (
    STOCKFISH_FIXED_NODES_ANCHOR_ELO,
    LadderRungObservation,
    LadderRungPairScores,
    bootstrap_ladder_elo_interval,
    fit_ladder_elo,
)


def _expected_score(anchor_elo: float, rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((anchor_elo - rating) / 400.0))


def _synthetic_observations(rating: float) -> tuple[LadderRungObservation, ...]:
    return tuple(
        LadderRungObservation(anchor_elo=anchor_elo, score=_expected_score(anchor_elo, rating), game_count=100)
        for anchor_elo in STOCKFISH_FIXED_NODES_ANCHOR_ELO.values()
    )


def test_agreed_anchor_curve_covers_the_four_plan_rungs() -> None:
    assert dict(STOCKFISH_FIXED_NODES_ANCHOR_ELO) == {30: 1100.0, 100: 1200.0, 300: 1400.0, 1000: 1700.0}


def test_logistic_fit_recovers_the_generating_rating() -> None:
    assert fit_ladder_elo(_synthetic_observations(1500.0)) == pytest.approx(1500.0, abs=0.1)


def test_single_even_rung_fits_to_its_anchor() -> None:
    observation = LadderRungObservation(anchor_elo=1400.0, score=0.5, game_count=100)

    assert fit_ladder_elo((observation,)) == pytest.approx(1400.0, abs=0.1)


def test_higher_rung_scores_fit_a_higher_elo() -> None:
    weaker = fit_ladder_elo(_synthetic_observations(1300.0))
    stronger = fit_ladder_elo(_synthetic_observations(1600.0))

    assert stronger > weaker


def test_fit_rejects_an_empty_ladder() -> None:
    with pytest.raises(ValueError, match='at least one rung'):
        fit_ladder_elo(())


def test_rung_level_bootstrap_collapses_for_perfectly_consistent_rungs() -> None:
    rating = 1450.0
    rungs = tuple(
        LadderRungPairScores(anchor_elo=anchor_elo, pair_scores=(_expected_score(anchor_elo, rating),) * 50)
        for anchor_elo in STOCKFISH_FIXED_NODES_ANCHOR_ELO.values()
    )

    low, high = bootstrap_ladder_elo_interval(rungs, 200, random_seed=7)

    assert low == pytest.approx(rating, abs=25.0)
    assert high == pytest.approx(rating, abs=25.0)


def test_rung_level_bootstrap_widens_when_rungs_disagree() -> None:
    consistent = tuple(
        LadderRungPairScores(anchor_elo=anchor_elo, pair_scores=(_expected_score(anchor_elo, 1450.0),) * 5)
        for anchor_elo in STOCKFISH_FIXED_NODES_ANCHOR_ELO.values()
    )
    disagreeing = (
        LadderRungPairScores(anchor_elo=1100.0, pair_scores=(0.0,) * 5),
        LadderRungPairScores(anchor_elo=1200.0, pair_scores=(1.0,) * 5),
        LadderRungPairScores(anchor_elo=1400.0, pair_scores=(0.0,) * 5),
        LadderRungPairScores(anchor_elo=1700.0, pair_scores=(1.0,) * 5),
    )

    consistent_low, consistent_high = bootstrap_ladder_elo_interval(consistent, 500, random_seed=11)
    disagreeing_low, disagreeing_high = bootstrap_ladder_elo_interval(disagreeing, 500, random_seed=11)

    assert disagreeing_high - disagreeing_low > consistent_high - consistent_low


def test_bootstrap_rejects_invalid_inputs() -> None:
    rung = LadderRungPairScores(anchor_elo=1400.0, pair_scores=(0.5,))
    with pytest.raises(ValueError, match='at least one rung'):
        bootstrap_ladder_elo_interval((), 10, random_seed=0)
    with pytest.raises(ValueError, match='sample count'):
        bootstrap_ladder_elo_interval((rung,), 0, random_seed=0)
    with pytest.raises(ValueError, match='lie in'):
        LadderRungPairScores(anchor_elo=1400.0, pair_scores=(1.5,))
    with pytest.raises(ValueError, match='game count'):
        LadderRungObservation(anchor_elo=1400.0, score=0.5, game_count=0)
