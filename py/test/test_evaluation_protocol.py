from pathlib import Path

import numpy as np
import pytest

from src.experiment.evaluation_protocol import (
    GameOutcome,
    GameRecord,
    PlayerColor,
    build_paired_schedule,
    load_opening_suite,
    summarize_match,
)
from src.eval.ModelEvaluationPy import _play_paired_models_search
from src.settings import CurrentBoard, CurrentGame


OPENING_SUITE_PATH = Path('reference/pilot-openings.tsv')


def test_opening_suite_builds_color_swapped_pairs() -> None:
    openings = load_opening_suite(OPENING_SUITE_PATH)

    schedule = build_paired_schedule(openings)

    assert len(openings) == 8
    assert len(schedule) == 16
    for pair_start in range(0, len(schedule), 2):
        white_game = schedule[pair_start]
        black_game = schedule[pair_start + 1]
        assert white_game.opening_id == black_game.opening_id
        assert white_game.fen == black_game.fen
        assert white_game.candidate_color == PlayerColor.WHITE
        assert black_game.candidate_color == PlayerColor.BLACK


def test_match_summary_uses_opening_pairs_for_bootstrap() -> None:
    records = (
        GameRecord(
            schedule_index=0,
            opening_id='a',
            starting_fen='fen-a',
            candidate_color=PlayerColor.WHITE,
            outcome=GameOutcome.WIN,
            moves_uci=(),
        ),
        GameRecord(
            schedule_index=1,
            opening_id='a',
            starting_fen='fen-a',
            candidate_color=PlayerColor.BLACK,
            outcome=GameOutcome.DRAW,
            moves_uci=(),
        ),
        GameRecord(
            schedule_index=2,
            opening_id='b',
            starting_fen='fen-b',
            candidate_color=PlayerColor.WHITE,
            outcome=GameOutcome.LOSS,
            moves_uci=(),
        ),
        GameRecord(
            schedule_index=3,
            opening_id='b',
            starting_fen='fen-b',
            candidate_color=PlayerColor.BLACK,
            outcome=GameOutcome.DRAW,
            moves_uci=(),
        ),
    )

    summary = summarize_match(records, bootstrap_seed=7, bootstrap_samples=2000)

    assert summary.wins == 1
    assert summary.draws == 2
    assert summary.losses == 1
    assert summary.score == pytest.approx(0.5)
    assert summary.opening_pair_count == 2
    assert summary.score_confidence_low == pytest.approx(0.25)
    assert summary.score_confidence_high == pytest.approx(0.75)
    assert summary.descriptive_logistic_elo_difference == pytest.approx(0)


def test_match_summary_rejects_unpaired_opening() -> None:
    records = (
        GameRecord(
            schedule_index=0,
            opening_id='a',
            starting_fen='fen-a',
            candidate_color=PlayerColor.WHITE,
            outcome=GameOutcome.WIN,
            moves_uci=(),
        ),
    )

    with pytest.raises(ValueError, match='exactly two games'):
        summarize_match(records, bootstrap_seed=7, bootstrap_samples=100)


def first_legal_move(boards: list[CurrentBoard]) -> list[np.ndarray]:
    return [CurrentGame.encode_moves([board.get_valid_moves()[0]], board) for board in boards]


def test_paired_match_reuses_opening_with_colors_swapped() -> None:
    opening = load_opening_suite(OPENING_SUITE_PATH)[0]
    schedule = build_paired_schedule((opening,))

    results, records = _play_paired_models_search(
        iteration=1,
        candidate_model=first_legal_move,
        opponent_model=first_legal_move,
        schedule=schedule,
        maximum_game_plies=4,
        name='test',
    )

    assert results.wins == 0
    assert results.draws == 2
    assert results.losses == 0
    assert records[0].starting_fen == records[1].starting_fen
    assert records[0].candidate_color == PlayerColor.WHITE
    assert records[1].candidate_color == PlayerColor.BLACK
    assert records[0].moves_uci == records[1].moves_uci
