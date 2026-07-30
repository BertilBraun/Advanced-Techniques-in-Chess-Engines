from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.az.games.go.augmentation import GoSymmetry, transform_action, transform_planes
from src.az.games.go.losses import calculate_go_loss
from src.az.games.go.model import ResidualGoModel
from src.az.games.go.replay_codec import GoReplayCodec
from src.az.games.go.samples import finalize_sample, pending_sample_from_native
from src.az.games.api import GameIdentifier
from src.az.replay.credits import ReplayCreditJournal
from src.az.replay.envelope import GameTermination, ReplayRecord
from src.az.replay.storage import ReplayShardStorage
from test.unit.go_stage5_helpers import (
    envelope,
    game_configuration,
    model_configuration,
    objective_configuration,
)

try:
    import az_go_native as native
except ImportError:
    native = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(native is None, reason='focused native Go extension has not been built'),
]


class UniformEvaluator:
    def __call__(self, request: native.GoInferenceRequest) -> native.InferenceResult:
        return native.InferenceResult(request.request_id, [1.0] * request.action_count, 0.0)


def fixed_configuration() -> native.FixedPuctConfiguration:
    return native.FixedPuctConfiguration(
        simulation_cap=4,
        exploration_constant=1.5,
        backup_discount=1.0,
        no_visited_child_value=0.0,
        action_temperature=0.0,
        root_noise_seed=123,
        action_sampling_seed=124,
        root_noise=native.RootNoiseConfiguration(enabled=False, alpha=0.3, fraction=0.25),
        tree_reuse=False,
    )


def test_native_search_to_replay_model_and_loss_cpu(tmp_path: Path) -> None:
    game = game_configuration()
    state = native.GoState(native.GoRules(7, 15, game.safety_ply_cap, game.history_length))
    result = native.search_go_fixed(state, UniformEvaluator(), fixed_configuration())
    pending = pending_sample_from_native(
        state.canonical_encoding(),
        tuple(state.legal_actions()),
        result,
        game,
    )
    finalized = finalize_sample(pending, 1.0, 1.0, GameTermination.TWO_CONSECUTIVE_PASSES)
    codec = GoReplayCodec(game, 1)
    replay = ReplayShardStorage(
        tmp_path,
        8,
        16,
        GameIdentifier.GO,
        1,
        'none',
        ReplayCreditJournal(tmp_path / 'credit-journal.bin'),
    )
    replay.publish(0, (ReplayRecord(envelope(), codec.encode(finalized)),))
    decoded = codec.decode(next(replay.records()).payload)
    batch = codec.create_batch((decoded,))
    torch.manual_seed(7)
    model = ResidualGoModel(game, model_configuration())

    loss = calculate_go_loss(model(batch.inputs), batch, model, objective_configuration())

    assert torch.isfinite(loss.total)
    loss.total.backward()


@pytest.mark.parametrize('symmetry', list(GoSymmetry))
def test_python_and_native_symmetries_agree(symmetry: GoSymmetry) -> None:
    game = game_configuration()
    state = native.GoState(native.GoRules(7, 15, game.safety_ply_cap, game.history_length))
    state.apply(0)
    encoding = state.canonical_encoding()
    native_symmetry = native.Symmetry(int(symmetry))

    transformed_native = native.transform_encoding(encoding, native_symmetry)
    transformed_python = transform_planes(
        np.asarray(encoding.values, dtype=np.float32).reshape(encoding.planes, 7, 7),
        symmetry,
    )

    assert np.array_equal(
        np.asarray(transformed_native.values).reshape(encoding.planes, 7, 7),
        transformed_python,
    )
    for action in range(game.action_count):
        assert native.transform_action(action, 7, native_symmetry) == transform_action(action, 7, symmetry)
