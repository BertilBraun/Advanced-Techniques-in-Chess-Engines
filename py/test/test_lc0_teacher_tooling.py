from __future__ import annotations

import random

import numpy as np
import pytest
from src.games.chess.contract import CHESS_BINARY_CHANNEL_COUNT, CHESS_CHANNEL_COUNT
from tools.build_lc0_policy_map import flip_uci_ranks
from tools.generate_lc0_policy_games import SamplingParameters, select_action, temperature_at, unpack_planes

SAMPLING = SamplingParameters(
    starting_temperature=1.3,
    final_temperature=0.1,
    greedy_after_ply=80,
    maximum_random_opening_plies=8,
    maximum_game_plies=300,
)


@pytest.mark.parametrize(
    ('move_uci', 'expected'),
    [
        ('e2e4', 'e7e5'),
        ('e7e5', 'e2e4'),
        ('a1h8', 'a8h1'),
        ('b7b8q', 'b2b1q'),
        ('g1f3', 'g8f6'),
    ],
)
def test_uci_rank_flip_is_an_involution(move_uci: str, expected: str) -> None:
    assert flip_uci_ranks(move_uci) == expected
    assert flip_uci_ranks(expected) == move_uci


def test_temperature_interpolates_from_start_to_final() -> None:
    assert temperature_at(0, SAMPLING) == pytest.approx(1.3)
    assert temperature_at(40, SAMPLING) == pytest.approx(0.7)
    assert temperature_at(80, SAMPLING) == pytest.approx(0.1)


def test_temperature_is_clamped_past_the_greedy_ply() -> None:
    assert temperature_at(500, SAMPLING) == pytest.approx(0.1)


def test_selection_is_greedy_after_the_greedy_ply() -> None:
    policy = np.array([0.1, 0.7, 0.2])
    chosen = {select_action([10, 11, 12], policy, 80, SAMPLING, random.Random(seed)) for seed in range(20)}
    assert chosen == {11}


def test_selection_samples_below_the_greedy_ply() -> None:
    policy = np.array([0.34, 0.33, 0.33])
    chosen = {select_action([10, 11, 12], policy, 0, SAMPLING, random.Random(seed)) for seed in range(40)}
    assert len(chosen) > 1


def test_selection_falls_back_to_argmax_on_a_degenerate_policy() -> None:
    policy = np.array([0.0, 0.0, 1.0])
    assert select_action([10, 11, 12], policy, 0, SAMPLING, random.Random(0)) == 12


def test_unpacked_planes_have_the_lc0_shape_and_dtype() -> None:
    scalar_count = CHESS_CHANNEL_COUNT - CHESS_BINARY_CHANNEL_COUNT
    payload = bytes(CHESS_BINARY_CHANNEL_COUNT * 8) + bytes([0, 0, 1])
    assert len(payload) == CHESS_BINARY_CHANNEL_COUNT * 8 + scalar_count
    planes = unpack_planes([payload, payload])
    assert planes.shape == (2, CHESS_CHANNEL_COUNT, 8, 8)
    assert planes.dtype.is_floating_point is False


def test_unpacked_constant_plane_is_broadcast_over_the_board() -> None:
    payload = bytes(CHESS_BINARY_CHANNEL_COUNT * 8) + bytes([0, 0, 1])
    planes = unpack_planes([payload])
    assert planes[0, CHESS_CHANNEL_COUNT - 1].min().item() == 1
    assert planes[0, CHESS_CHANNEL_COUNT - 1].max().item() == 1
    assert planes[0, CHESS_CHANNEL_COUNT - 2].max().item() == 0
