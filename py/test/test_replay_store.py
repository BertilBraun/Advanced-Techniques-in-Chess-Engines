from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import GameSearchVisit
from src.games.contracts import WdlTarget
from src.games.representation import PackedPlaneLayout
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    IneligibleNextPolicyTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout, TrainingTargetLayout


def _layout(ply_offset: int = 1) -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PackedPlaneLayout(board_size=3, binary_plane_count=2, scalar_count=1),
        targets=TrainingTargetLayout(
            action_size=10,
            wdl_size=3,
            auxiliary_heads=(
                NextPolicyHeadLayout(kind='next_policy', action_size=10, ply_offset=ply_offset),
                RemainingGameLengthHeadLayout(kind='remaining_game_length', normalization_scale=20.0),
            ),
        ),
        maximum_policy_entries=4,
        maximum_legal_actions=10,
    )


def _sample(layout: ReplayLayout, action_id: int, generation: int, auxiliary_eligible: bool = True) -> ReplaySample:
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts.from_native(
            (
                GameSearchVisit(action_id=action_id, visit_count=7),
                GameSearchVisit(action_id=(action_id + 1) % 10, visit_count=3),
            )
        ),
        legal_action_ids=tuple(range(10)),
    )
    auxiliary = EligibleNextPolicyTarget(policy=policy) if auxiliary_eligible else IneligibleNextPolicyTarget()
    return ReplaySample(
        encoded_state=layout.packed_planes.value(bytes([action_id]) * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=0.125,
        auxiliary_targets=(
            auxiliary,
            EligibleRemainingGameLengthTarget(normalized_length=(action_id + 1) / 20),
        ),
        sample_weight=1.5,
        source_model_generation=generation,
        source_created_at_seconds=123.0 + generation,
    )


def test_replay_store_persists_fixed_rows_and_fifo_state(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, layout, maximum_capacity=5, logical_capacity=3)
    assert store.allocated_file_size == ReplayStore.projected_file_size(layout, 5)
    samples = tuple(
        _sample(layout, action_id, action_id, auxiliary_eligible=action_id % 2 == 0) for action_id in range(5)
    )

    for sample in samples:
        store.append(sample)
    store.flush()

    assert store.state.size == 3
    assert store.state.evicted_rows == 2
    assert tuple(store.sample_at(index).source_model_generation for index in range(3)) == (2, 3, 4)
    assert store.sample_at(0).policy == samples[2].policy
    stored_auxiliary = store.sample_at(0).auxiliary_targets
    assert stored_auxiliary[0] == samples[2].auxiliary_targets[0]
    assert isinstance(stored_auxiliary[1], EligibleRemainingGameLengthTarget)
    assert stored_auxiliary[1].normalized_length == pytest.approx(0.15)
    store.close()

    reopened = ReplayStore.open(path, layout)
    assert reopened.state.size == 3
    assert tuple(reopened.sample_at(index).source_model_generation for index in range(3)) == (2, 3, 4)
    reopened.set_logical_capacity(2)
    assert reopened.state.size == 2
    assert reopened.state.evicted_rows == 3
    assert tuple(reopened.sample_at(index).source_model_generation for index in range(2)) == (3, 4)
    reopened.close()


def test_replay_store_rejects_layout_with_same_width_but_different_target_semantics(tmp_path: Path) -> None:
    path = tmp_path / 'replay.bin'
    store = ReplayStore.create(path, _layout(ply_offset=1), maximum_capacity=2, logical_capacity=2)
    store.close()

    with pytest.raises(ValueError, match='layout'):
        ReplayStore.open(path, _layout(ply_offset=2))


def test_replay_store_bulk_extend_matches_individual_appends_across_wraparound(tmp_path: Path) -> None:
    layout = _layout()
    samples = tuple(_sample(layout, action_id % 9, action_id) for action_id in range(8))
    appended = ReplayStore.create(tmp_path / 'appended.bin', layout, maximum_capacity=5, logical_capacity=3)
    extended = ReplayStore.create(tmp_path / 'extended.bin', layout, maximum_capacity=5, logical_capacity=3)

    for sample in samples:
        appended.append(sample)
    extended.extend(samples)

    assert extended.state == appended.state
    assert tuple(extended.sample_at(index) for index in range(3)) == tuple(
        appended.sample_at(index) for index in range(3)
    )
    appended.close()
    extended.close()


def test_replay_store_rejects_sparse_policy_beyond_fixed_width(tmp_path: Path) -> None:
    layout = _layout()
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=2, logical_capacity=2)
    sample = _sample(layout, 0, 0)
    oversized_policy = SparsePolicyTarget(
        visits=SearchVisitCounts.from_native(
            tuple(GameSearchVisit(action_id=action_id, visit_count=1) for action_id in range(5))
        ),
        legal_action_ids=tuple(range(10)),
    )
    oversized = ReplaySample(
        encoded_state=sample.encoded_state,
        policy=oversized_policy,
        wdl_target=sample.wdl_target,
        root_value=sample.root_value,
        auxiliary_targets=sample.auxiliary_targets,
        sample_weight=sample.sample_weight,
        source_model_generation=sample.source_model_generation,
        source_created_at_seconds=sample.source_created_at_seconds,
    )

    with pytest.raises(ValueError, match='retained-entry'):
        store.append(oversized)
    store.close()
