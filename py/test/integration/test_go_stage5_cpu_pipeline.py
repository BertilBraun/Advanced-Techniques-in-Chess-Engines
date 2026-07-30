from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from src.az.games.go.losses import calculate_go_loss
from src.az.games.go.model import ResidualGoModel
from src.az.games.go.replay_codec import GoReplayCodec
from src.az.games.go.samples import finalize_sample, pending_sample_from_native
from src.az.games.api import GameIdentifier
from src.az.replay.envelope import GameTermination, ReplayRecord
from src.az.replay.storage import ReplayShardStorage
from test.unit.go_stage5_helpers import (
    envelope,
    game_configuration,
    model_configuration,
    objective_configuration,
)

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class FixedSearchEncoding:
    planes: int
    board_size: int
    values: list[int]


@dataclass(frozen=True)
class FixedSearchTelemetry:
    policy_target_eligible: bool
    policy_target_weight: float


@dataclass(frozen=True)
class FixedSearchResult:
    root_visits: list[int]
    telemetry: FixedSearchTelemetry


def fixed_search_fixture() -> tuple[FixedSearchEncoding, FixedSearchResult]:
    game = game_configuration()
    encoding = FixedSearchEncoding(
        planes=game.input_plane_count,
        board_size=game.board_size,
        values=[0] * (game.input_plane_count * game.board_size**2),
    )
    result = FixedSearchResult(
        root_visits=[4] + [0] * (game.action_count - 1),
        telemetry=FixedSearchTelemetry(policy_target_eligible=True, policy_target_weight=1),
    )
    return encoding, result


def test_fixed_search_fixture_round_trips_to_cpu_loss(tmp_path: Path) -> None:
    game = game_configuration()
    encoding, result = fixed_search_fixture()
    pending = pending_sample_from_native(
        encoding,
        tuple(range(game.action_count)),
        result,
        game,
    )
    sample = finalize_sample(pending, 1, 1, GameTermination.TWO_CONSECUTIVE_PASSES)
    codec = GoReplayCodec(game, 1)
    storage = ReplayShardStorage(tmp_path, 8, 16, GameIdentifier.GO, 1, 'none')
    storage.publish(0, (ReplayRecord(envelope(), codec.encode(sample)),))
    decoded = codec.decode(next(storage.records()).payload)
    batch = codec.create_batch((decoded,))
    torch.manual_seed(11)
    model = ResidualGoModel(game, model_configuration())

    result_loss = calculate_go_loss(model(batch.inputs), batch, model, objective_configuration())
    result_loss.total.backward()

    assert torch.isfinite(result_loss.total)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_capped_fixed_search_fixture_has_no_value_loss() -> None:
    game = game_configuration()
    encoding, result = fixed_search_fixture()
    pending = pending_sample_from_native(encoding, tuple(range(game.action_count)), result, game)

    sample = finalize_sample(pending, None, 1, GameTermination.SAFETY_PLY_CAP)
    batch = GoReplayCodec(game, 1).create_batch((sample,))
    model = ResidualGoModel(game, model_configuration())
    result_loss = calculate_go_loss(model(batch.inputs), batch, model, objective_configuration())

    assert sample.value_target is None
    assert result_loss.value.eligible_count == 0
    assert result_loss.value.weighted_sum.item() == 0


def test_native_conversion_rejects_duplicate_legal_actions() -> None:
    game = game_configuration()
    encoding, result = fixed_search_fixture()

    with pytest.raises(ValueError, match='duplicate'):
        pending_sample_from_native(encoding, (0, 0, 1), result, game)
