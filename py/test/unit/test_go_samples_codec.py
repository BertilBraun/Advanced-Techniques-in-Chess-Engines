from __future__ import annotations

import numpy as np
import pytest

from src.az.games.go.replay_codec import GoReplayCodec
from src.az.games.go.samples import (
    DensePolicyTarget,
    GoSample,
    PendingGoSearchSample,
    SparsePolicyTarget,
    create_batch,
    finalize_sample,
)
from src.az.replay.envelope import GameTermination
from test.unit.go_stage5_helpers import game_configuration, sample


@pytest.mark.parametrize('sparse', [False, True])
def test_go_payload_round_trip_is_deterministic(sparse: bool) -> None:
    configuration = game_configuration()
    original = sample()
    if sparse:
        original = GoSample(
            input_planes=original.input_planes,
            legal_action_mask=original.legal_action_mask,
            policy_target=SparsePolicyTarget(
                actions=np.asarray([0, 4, configuration.action_count - 1], dtype=np.int32),
                weights=np.asarray([2, 3, 5], dtype=np.float32),
            ),
            policy_weight=0.5,
            value_target=-1.0,
            value_weight=2.0,
        )
    codec = GoReplayCodec(configuration, payload_schema_version=1)

    first_encoding = codec.encode(original)
    decoded = codec.decode(first_encoding)

    assert codec.encode(decoded) == first_encoding
    assert np.array_equal(decoded.input_planes, original.input_planes)
    assert np.array_equal(decoded.legal_action_mask, original.legal_action_mask)
    assert decoded.policy_weight == pytest.approx(original.policy_weight)
    assert decoded.value_target == pytest.approx(original.value_target)
    assert decoded.value_weight == pytest.approx(original.value_weight)


def test_go_codec_rejects_corruption_schema_and_shape() -> None:
    configuration = game_configuration()
    codec = GoReplayCodec(configuration, 1)
    encoded = codec.encode(sample())

    with pytest.raises(ValueError, match='checksum'):
        codec.decode(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    with pytest.raises(ValueError, match='truncated'):
        codec.decode(encoded[:20])
    with pytest.raises(ValueError, match='schema'):
        GoReplayCodec(configuration, 2)
    with pytest.raises(ValueError, match='shape'):
        GoReplayCodec(game_configuration(9), 1).decode(encoded)


@pytest.mark.parametrize('history_length', [256, 1024])
def test_go_codec_supports_full_configured_history_range(history_length: int) -> None:
    configuration = game_configuration(history_length=history_length)
    original = sample(history_length=history_length)
    codec = GoReplayCodec(configuration, 1)

    decoded = codec.decode(codec.encode(original))

    assert decoded.input_planes.shape == (history_length * 2 + 1, 7, 7)


def test_sample_validates_policy_shape_legality_and_value_weight() -> None:
    original = sample()
    illegal = original.legal_action_mask.copy()
    illegal[0] = False

    with pytest.raises(ValueError, match='illegal action'):
        GoSample(
            input_planes=original.input_planes,
            legal_action_mask=illegal,
            policy_target=original.policy_target,
            policy_weight=1,
            value_target=1,
            value_weight=1,
        )
    with pytest.raises(ValueError, match='exactly when'):
        GoSample(
            input_planes=original.input_planes,
            legal_action_mask=original.legal_action_mask,
            policy_target=original.policy_target,
            policy_weight=1,
            value_target=None,
            value_weight=1,
        )


def test_censored_finalization_keeps_policy_and_removes_value() -> None:
    original = sample()
    pending = PendingGoSearchSample(
        original.input_planes,
        original.legal_action_mask,
        original.policy_target,
        original.policy_weight,
    )

    censored = finalize_sample(
        pending, value_target=None, configured_value_weight=1, termination=GameTermination.SAFETY_PLY_CAP
    )

    assert censored.policy_weight == 1
    assert censored.value_target is None
    assert censored.value_weight == 0


def test_batch_normalizes_dense_and_sparse_targets() -> None:
    configuration = game_configuration()
    dense = sample()
    sparse = GoSample(
        input_planes=dense.input_planes,
        legal_action_mask=dense.legal_action_mask,
        policy_target=SparsePolicyTarget(
            np.asarray([0, configuration.action_count - 1], dtype=np.int32),
            np.asarray([1, 3], dtype=np.float32),
        ),
        policy_weight=1,
        value_target=None,
        value_weight=0,
    )

    batch = create_batch((dense, sparse), configuration)

    assert batch.inputs.shape == (2, configuration.input_plane_count, 7, 7)
    assert batch.policy_targets.sum(dim=1).tolist() == pytest.approx([1, 1])
    assert batch.policy_targets[1, 0].item() == pytest.approx(0.25)
    assert batch.policy_targets[1, -1].item() == pytest.approx(0.75)
    assert batch.value_weights.tolist() == [1, 0]


def test_zero_weight_policy_may_have_zero_mass() -> None:
    configuration = game_configuration()
    original = sample()
    zero_policy = GoSample(
        input_planes=original.input_planes,
        legal_action_mask=original.legal_action_mask,
        policy_target=DensePolicyTarget(np.zeros(configuration.action_count, dtype=np.float32)),
        policy_weight=0,
        value_target=0,
        value_weight=1,
    )

    batch = create_batch((zero_policy,), configuration)

    assert batch.policy_targets.sum().item() == 0


def test_large_finite_policy_entries_normalize_without_overflow() -> None:
    configuration = game_configuration()
    original = sample()
    probabilities = np.full(configuration.action_count, np.finfo(np.float32).max, dtype=np.float32)
    large_target = GoSample(
        input_planes=original.input_planes,
        legal_action_mask=original.legal_action_mask,
        policy_target=DensePolicyTarget(probabilities),
        policy_weight=1,
        value_target=0,
        value_weight=1,
    )

    batch = create_batch((large_target,), configuration)

    assert np.isfinite(batch.policy_targets.numpy()).all()
    assert batch.policy_targets.sum().item() == pytest.approx(1)
