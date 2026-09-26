from __future__ import annotations

import pytest
import torch
from src.distillation.dataset import CHESS_PAYLOAD_BYTES, DistillationRecordLayout, record_dtype
from src.distillation.lc0_teacher import Lc0TeacherNetwork, sampling_temperature_at
from src.games.chess.contract import CHESS_BINARY_CHANNEL_COUNT, CHESS_CHANNEL_COUNT, CHESS_STATE_CONTRACT
from tools.build_lc0_policy_map import flip_uci_ranks, lc0_move_notation
from tools.build_lc0_teacher_model import Lc0TeacherModel

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
    output = network.training_output(torch.zeros((2, CHESS_CHANNEL_COUNT, 8, 8)))
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
    policy, _ = wrapped_constant_model()(torch.zeros((1, CHESS_CHANNEL_COUNT, 8, 8), dtype=torch.int8))
    assert policy[0, 3].item() == pytest.approx(0.0)
    assert policy[0, 1].item() == pytest.approx(1.0)


def test_actions_sharing_an_lc0_index_read_the_same_logit() -> None:
    policy, _ = wrapped_constant_model()(torch.zeros((1, CHESS_CHANNEL_COUNT, 8, 8), dtype=torch.int8))
    assert policy[0, 1].item() == policy[0, 2].item()


def test_unmapped_actions_stay_finite_and_negligible() -> None:
    policy, _ = wrapped_constant_model()(torch.zeros((1, CHESS_CHANNEL_COUNT, 8, 8), dtype=torch.int8))
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
