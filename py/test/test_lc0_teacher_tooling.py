from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src.distillation.dataset import (
    CHESS_PAYLOAD_BYTES,
    DistillationDatasetManifest,
    DistillationRecordLayout,
    open_dataset,
    record_dtype,
    write_dataset,
)
from src.distillation.lc0_teacher import Lc0TeacherNetwork, sampling_temperature_at
from src.distillation.stockfish_moves import (
    ScoredMove,
    UciCandidate,
    expected_score,
    parse_multipv_line,
    sample_scored_move,
)
from src.games.chess.contract import (
    CHESS_BINARY_CHANNEL_COUNT,
    CHESS_CHANNEL_COUNT,
    CHESS_STATE_CONTRACT,
    LC0_CHANNEL_COUNT,
)
from tools.build_lc0_policy_map import flip_uci_ranks, lc0_move_notation
from tools.build_lc0_teacher_model import Lc0TeacherModel
from tools.distill_merge_datasets import merge_datasets

STARTING_TEMPERATURE = 1.3
FINAL_TEMPERATURE = 0.1


def temperature(ply: int, greedy_after_ply: int | None) -> float:
    return sampling_temperature_at(ply, STARTING_TEMPERATURE, FINAL_TEMPERATURE, greedy_after_ply)


class ConstantBackbone(torch.nn.Module):
    def __init__(self, policy_size: int, wdl: tuple[float, float, float]) -> None:
        super().__init__()
        self.policy_size = policy_size
        self.wdl = wdl

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = planes.shape[0]
        policy = torch.arange(self.policy_size, dtype=torch.float32).expand(batch, -1)
        return policy, torch.tensor([self.wdl], dtype=torch.float32).expand(batch, -1)


@pytest.mark.parametrize(
    ('move_uci', 'expected'),
    [('e2e4', 'e7e5'), ('e7e5', 'e2e4'), ('a1h8', 'a8h1'), ('b7b8q', 'b2b1q'), ('g1f3', 'g8f6')],
)
def test_uci_rank_flip_is_an_involution(move_uci: str, expected: str) -> None:
    assert flip_uci_ranks(move_uci) == expected
    assert flip_uci_ranks(expected) == move_uci


def test_temperature_interpolates_from_start_to_final() -> None:
    assert temperature(0, greedy_after_ply=80) == pytest.approx(1.3)
    assert temperature(40, greedy_after_ply=80) == pytest.approx(0.7)
    assert temperature(80, greedy_after_ply=80) == pytest.approx(0.1)


def test_temperature_is_clamped_past_the_greedy_ply() -> None:
    assert temperature(500, greedy_after_ply=80) == pytest.approx(0.1)


def test_temperature_is_fixed_without_a_greedy_ply() -> None:
    assert temperature(0, greedy_after_ply=None) == pytest.approx(1.3)
    assert temperature(200, greedy_after_ply=None) == pytest.approx(1.3)


def test_teacher_wdl_survives_the_callers_softmax() -> None:
    probabilities = (0.6, 0.25, 0.15)
    network = Lc0TeacherNetwork(ConstantBackbone(policy_size=1880, wdl=probabilities))
    output = network.training_output(torch.zeros((2, LC0_CHANNEL_COUNT, 8, 8)))
    recovered = torch.softmax(output.wdl_logits, dim=1)
    assert recovered[0].tolist() == pytest.approx(list(probabilities), abs=1e-6)


def wrapped_constant_model() -> Lc0TeacherModel:
    # Action 0 is unmapped (index 3 is the sentinel past a 3-wide policy); actions 1 and 2 share index 1,
    # as a knight promotion and a plain move to the same square do in Lc0's table.
    permutation = torch.tensor([3, 1, 1, 0], dtype=torch.int64)
    backbone = ConstantBackbone(policy_size=3, wdl=(1.0, 0.0, 0.0))
    return Lc0TeacherModel(
        backbone, permutation, wdl_is_already_probability=True, policy_output_index=0, wdl_output_index=1
    )


def test_each_action_reads_its_lc0_logit() -> None:
    policy, _ = wrapped_constant_model()(torch.zeros((1, LC0_CHANNEL_COUNT, 8, 8)))
    assert policy[0, 3].item() == pytest.approx(0.0)
    assert policy[0, 1].item() == pytest.approx(1.0)


def test_actions_sharing_an_lc0_index_read_the_same_logit() -> None:
    policy, _ = wrapped_constant_model()(torch.zeros((1, LC0_CHANNEL_COUNT, 8, 8)))
    assert policy[0, 1].item() == policy[0, 2].item()


def test_unmapped_actions_stay_finite_and_negligible() -> None:
    policy, _ = wrapped_constant_model()(torch.zeros((1, LC0_CHANNEL_COUNT, 8, 8)))
    assert torch.isfinite(policy).all().item()
    assert policy[0, 0].item() < -1.0e3


def test_payload_bytes_track_the_packed_layout() -> None:
    layout = CHESS_STATE_CONTRACT.packed_plane_layout
    assert CHESS_PAYLOAD_BYTES == CHESS_BINARY_CHANNEL_COUNT * 8 + (CHESS_CHANNEL_COUNT - CHESS_BINARY_CHANNEL_COUNT)
    assert CHESS_PAYLOAD_BYTES == layout.payload_bytes


def test_core_records_carry_no_auxiliary_fields() -> None:
    core = record_dtype(CHESS_PAYLOAD_BYTES, DistillationRecordLayout.CORE)
    assert 'next_policy_count' not in core.names
    assert 'wdl' in core.names


@pytest.mark.parametrize(
    ('fen', 'move_uci', 'expected'),
    [
        ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1', 'e1h1'),
        ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1c1', 'e1a1'),
        ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8g8', 'e8h8'),
        ('8/P6k/8/8/8/8/8/K7 w - - 0 1', 'a7a8n', 'a7a8'),
        ('8/P6k/8/8/8/8/8/K7 w - - 0 1', 'a7a8q', 'a7a8q'),
        ('8/8/8/8/8/8/8/K6k w - - 0 1', 'a1a2', 'a1a2'),
    ],
)
def test_moves_are_spelled_as_lc0_indexes_them(fen: str, move_uci: str, expected: str) -> None:
    assert lc0_move_notation(fen, move_uci) == expected


def test_the_wrapper_answers_any_batch_up_to_the_fixed_size() -> None:
    small, _ = wrapped_constant_model()(torch.zeros((3, LC0_CHANNEL_COUNT, 8, 8)))
    full, _ = wrapped_constant_model()(torch.zeros((64, LC0_CHANNEL_COUNT, 8, 8)))
    assert small.shape[0] == 3 and full.shape[0] == 64


def test_the_teacher_adapter_chunks_batches_beyond_the_fixed_size() -> None:
    network = Lc0TeacherNetwork(ConstantBackbone(policy_size=1880, wdl=(0.5, 0.3, 0.2)))
    output = network.training_output(torch.zeros((150, LC0_CHANNEL_COUNT, 8, 8)))
    assert output.policy_logits.shape[0] == 150


def core_part(directory: Path, name: str, seed: int, rows: int) -> Path:
    path = directory / f'{name}.bin'
    records = np.zeros(rows, dtype=record_dtype(CHESS_PAYLOAD_BYTES, DistillationRecordLayout.CORE))
    records['legal_count'] = seed
    manifest = DistillationDatasetManifest(
        game='chess',
        position_count=rows,
        action_size=1880,
        payload_bytes=CHESS_PAYLOAD_BYTES,
        maximum_policy_entries=64,
        maximum_legal_actions=218,
        teacher_generation=0,
        teacher_weights_sha256='0' * 64,
        teacher_parameter_count=1,
        random_seed=seed,
        random_opening_plies=8,
        sampling_temperature=1.3,
        sample_one_position_in=1,
        random_perturbation_probability=0.0,
        maximum_game_plies=300,
        builder_source_revision='test',
        record_layout=DistillationRecordLayout.CORE,
    )
    write_dataset(path, records, manifest)
    return path


def test_core_layout_parts_merge_to_their_combined_rows(tmp_path: Path) -> None:
    parts = (core_part(tmp_path, 'a', 1, 3), core_part(tmp_path, 'b', 2, 5))
    merged = merge_datasets(parts, tmp_path / 'merged.bin')
    records, _ = open_dataset(tmp_path / 'merged.bin')
    assert merged.position_count == 8
    assert records['legal_count'].tolist() == [1, 1, 1, 2, 2, 2, 2, 2]


def test_merge_can_delete_each_part_once_copied(tmp_path: Path) -> None:
    parts = (core_part(tmp_path, 'a', 1, 3), core_part(tmp_path, 'b', 2, 5))
    merge_datasets(parts, tmp_path / 'merged.bin', delete_inputs=True)
    assert not any(part.exists() for part in parts)


STOCKFISH_CANDIDATES = (ScoredMove(10, 0.60), ScoredMove(11, 0.57), ScoredMove(12, 0.30))


def test_near_equal_stockfish_moves_are_both_played() -> None:
    generator = np.random.default_rng(0)
    chosen = {sample_scored_move(STOCKFISH_CANDIDATES, 0.05, generator) for _ in range(200)}
    assert {10, 11} <= chosen


def test_clearly_worse_stockfish_moves_are_practically_never_played() -> None:
    generator = np.random.default_rng(0)
    chosen = [sample_scored_move(STOCKFISH_CANDIDATES, 0.05, generator) for _ in range(2000)]
    # A move 0.30 worse weighs e^-6 of the best: about 0.2% of choices, never a routine event.
    assert chosen.count(12) / len(chosen) < 0.01


@pytest.mark.parametrize(
    ('line', 'expected'),
    [
        (
            'info depth 9 seldepth 12 multipv 2 score cp -35 nodes 1000 nps 900000 pv e7e5 g1f3',
            (2, UciCandidate('e7e5', 'cp', -35)),
        ),
        (
            'info depth 8 multipv 1 score cp 20 lowerbound nodes 700 pv d2d4',
            (1, UciCandidate('d2d4', 'cp', 20)),
        ),
        ('info depth 12 multipv 3 score mate -2 nodes 1000 pv h7h8q', (3, UciCandidate('h7h8q', 'mate', -2))),
        ('info depth 5 currmove e2e4 currmovenumber 1', None),
        ('bestmove e2e4 ponder e7e5', None),
    ],
)
def test_multipv_lines_yield_their_index_move_and_score(line: str, expected: object) -> None:
    assert parse_multipv_line(line) == expected


START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def test_a_level_score_is_an_even_expected_score() -> None:
    assert expected_score(UciCandidate('e2e4', 'cp', 0), START_FEN) == pytest.approx(0.5, abs=0.01)


def test_a_centipawn_advantage_raises_the_expected_score() -> None:
    assert expected_score(UciCandidate('e2e4', 'cp', 150), START_FEN) > 0.6


@pytest.mark.parametrize(('mate', 'expected'), [(3, 1.0), (-3, 0.0)])
def test_mate_scores_are_certain_results(mate: int, expected: float) -> None:
    assert expected_score(UciCandidate('e2e4', 'mate', mate), START_FEN) == pytest.approx(expected)
