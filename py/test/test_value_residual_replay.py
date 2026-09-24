from __future__ import annotations

from pathlib import Path

import pytest
from src.games.contracts import WdlTarget
from src.games.representation import PackedPlaneLayout
from src.replay.contracts import (
    EligibleScalarAuxiliaryTarget,
    IneligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.encoding import encode_replay_columns
from src.replay.layout import ReplayLayout
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchVisitCounts
from src.training.targets import TrainingTargetLayout, ValueResidualHeadLayout


def _layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=PackedPlaneLayout(board_size=3, binary_plane_count=2, scalar_count=1),
        targets=TrainingTargetLayout(
            action_size=10,
            wdl_size=3,
            auxiliary_heads=(ValueResidualHeadLayout(kind='value_residual', smooth_l1_beta=0.05),),
        ),
        maximum_policy_entries=4,
        maximum_legal_actions=10,
    )


def _sample(layout: ReplayLayout, residual: float | None) -> ReplaySample:
    policy = SparsePolicyTarget(
        visits=SearchVisitCounts(action_ids=(2, 4), visit_counts=(6, 2)),
        legal_action_ids=(1, 2, 4, 7),
    )
    target = (
        IneligibleScalarAuxiliaryTarget(kind='value_residual')
        if residual is None
        else EligibleScalarAuxiliaryTarget(kind='value_residual', value=residual)
    )
    return ReplaySample(
        encoded_state=layout.packed_planes.value(b'\x01' * layout.packed_planes.payload_bytes),
        policy=policy,
        wdl_target=WdlTarget(win=0.5, draw=0.25, loss=0.25),
        root_value=-0.5,
        auxiliary_targets=(target,),
        sample_weight=1.0,
        policy_surprise=0.25,
        source_model_generation=3,
        source_created_at_seconds=1.0,
    )


def test_value_residual_head_allocates_its_replay_columns() -> None:
    names = {descriptor.key.name for descriptor in _layout().columns.columns}

    assert 'auxiliary_0_value' in names
    assert 'auxiliary_0_eligible' in names


def test_value_residual_survives_a_store_round_trip(tmp_path: Path) -> None:
    layout = _layout()
    sample = _sample(layout, 0.375)
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=4, logical_capacity=4)
    try:
        store.append(sample)

        assert store.sample_at(0) == sample
    finally:
        store.close()


def test_an_ineligible_value_residual_survives_a_store_round_trip(tmp_path: Path) -> None:
    layout = _layout()
    sample = _sample(layout, None)
    store = ReplayStore.create(tmp_path / 'replay.bin', layout, maximum_capacity=4, logical_capacity=4)
    try:
        store.append(sample)

        assert store.sample_at(0) == sample
    finally:
        store.close()


def test_value_residual_columns_encode_without_falling_through_the_layout_match() -> None:
    layout = _layout()

    columns = encode_replay_columns(layout, (_sample(layout, 0.375),))

    assert columns.auxiliary[0].kind == 'value_residual'
    assert columns.auxiliary[0].value[0] == pytest.approx(0.375)
    assert int(columns.auxiliary[0].eligible[0]) == 1
