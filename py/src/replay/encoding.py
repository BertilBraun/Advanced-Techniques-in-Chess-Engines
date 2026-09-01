from __future__ import annotations

import numpy as np
import numpy.typing as npt
from src.replay.columnar import (
    ReplayColumnArray,
    ReplayColumnViews,
    ReplayLegalMovesColumnViews,
    ReplayNextPolicyColumnViews,
    ReplayPolicyColumnViews,
    ReplayScalarColumnViews,
    build_column_views,
    flatten_column_views,
)
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    IneligibleNextPolicyTarget,
    IneligibleRemainingGameLengthTarget,
    IneligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
)


def encode_replay_columns(layout: ReplayLayout, samples: tuple[ReplaySample, ...]) -> ReplayColumnViews:
    row_count = len(samples)
    arrays = tuple(
        ReplayColumnArray(
            descriptor,
            np.zeros(
                (row_count, *descriptor.trailing_shape),
                dtype=descriptor.element_type.numpy_dtype,
            ),
        )
        for descriptor in layout.columns.columns
    )
    columns = build_column_views(layout, arrays)
    for row_index, sample in enumerate(samples):
        _encode_sample(layout, columns, row_index, sample)
    return columns


def encode_replay_rows(layout: ReplayLayout, samples: tuple[ReplaySample, ...]) -> npt.NDArray[np.void]:
    columns = encode_replay_columns(layout, samples)
    encoded_rows = np.zeros((len(samples),), dtype=layout.row_dtype)
    for column in flatten_column_views(layout, columns):
        encoded_rows[column.descriptor.key.name] = column.values
    return encoded_rows


def _encode_sample(
    layout: ReplayLayout,
    columns: ReplayColumnViews,
    row_index: int,
    sample: ReplaySample,
) -> None:
    if len(sample.encoded_state) != layout.packed_planes.payload_bytes:
        raise ValueError('Replay sample packed state has the wrong width.')
    if len(sample.auxiliary_targets) != len(layout.targets.auxiliary_heads):
        raise ValueError('Replay sample auxiliary targets do not match the fixed layout.')
    columns.encoded_state[row_index] = np.frombuffer(bytes(sample.encoded_state), dtype=np.uint8)
    _encode_policy(layout, columns.policy, row_index, sample.policy)
    columns.wdl_target[row_index] = (sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss)
    columns.root_value[row_index] = sample.root_value
    for head, target, destination in zip(
        layout.targets.auxiliary_heads,
        sample.auxiliary_targets,
        columns.auxiliary,
        strict=True,
    ):
        match head, target, destination:
            case NextPolicyHeadLayout(), EligibleNextPolicyTarget(policy=policy), ReplayNextPolicyColumnViews():
                _encode_policy(layout, destination.policy, row_index, policy)
                destination.eligible[row_index] = 1
            case NextPolicyHeadLayout(), IneligibleNextPolicyTarget(), ReplayNextPolicyColumnViews():
                destination.eligible[row_index] = 0
            case (
                RemainingGameLengthHeadLayout(),
                EligibleRemainingGameLengthTarget(normalized_length=value),
                ReplayScalarColumnViews(kind='remaining_game_length'),
            ):
                destination.value[row_index] = value
                destination.eligible[row_index] = 1
            case (
                RemainingGameLengthHeadLayout(),
                IneligibleRemainingGameLengthTarget(),
                ReplayScalarColumnViews(kind='remaining_game_length'),
            ):
                destination.eligible[row_index] = 0
            case (
                FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout(),
                EligibleScalarAuxiliaryTarget(value=value),
                ReplayScalarColumnViews(),
            ):
                destination.value[row_index] = value
                destination.eligible[row_index] = 1
            case (
                FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout(),
                IneligibleScalarAuxiliaryTarget(),
                ReplayScalarColumnViews(),
            ):
                destination.eligible[row_index] = 0
            case LegalMovesHeadLayout(), EligibleLegalMovesTarget(), ReplayLegalMovesColumnViews():
                pass
            case _:
                raise ValueError('Replay auxiliary target does not match its fixed layout.')
    columns.sample_weight[row_index] = sample.sample_weight
    columns.source_model_generation[row_index] = sample.source_model_generation
    columns.source_timestamp[row_index] = sample.source_created_at_seconds


def _encode_policy(
    layout: ReplayLayout,
    destination: ReplayPolicyColumnViews,
    row_index: int,
    policy: SparsePolicyTarget,
) -> None:
    if len(policy.visits.action_ids) > layout.maximum_policy_entries:
        raise ValueError('Sparse policy exceeds the configured retained-entry count.')
    if any(action_id >= layout.targets.action_size for action_id in policy.visits.action_ids):
        raise ValueError('Sparse policy contains an action outside the action space.')
    if any(visit_count > 65_535 for visit_count in policy.visits.visit_counts):
        raise ValueError('Sparse policy visit count does not fit uint16.')
    if len(policy.legal_action_ids) > layout.maximum_legal_actions:
        raise ValueError('Sparse policy exceeds the game maximum legal-action count.')
    if any(action_id >= layout.targets.action_size for action_id in policy.legal_action_ids):
        raise ValueError('Sparse policy contains a legal action outside the action space.')
    entry_count = len(policy.visits.action_ids)
    destination.entry_count[row_index] = entry_count
    destination.action_ids[row_index, :entry_count] = policy.visits.action_ids
    destination.visit_counts[row_index, :entry_count] = policy.visits.visit_counts
    legal_count = len(policy.legal_action_ids)
    destination.legal_count[row_index] = legal_count
    destination.legal_action_ids[row_index, :legal_count] = policy.legal_action_ids
