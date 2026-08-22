from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from src.replay.layout import ReplayColumnDescriptor, ReplayColumnKey, ReplayColumnKind, ReplayLayout
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
)

ReplayArray: TypeAlias = (
    npt.NDArray[np.uint8]
    | npt.NDArray[np.uint16]
    | npt.NDArray[np.uint32]
    | npt.NDArray[np.float32]
    | npt.NDArray[np.float64]
)


@dataclass(frozen=True)
class ReplayColumnArray:
    descriptor: ReplayColumnDescriptor
    values: ReplayArray


@dataclass(frozen=True)
class ReplayPolicyColumnViews:
    entry_count: npt.NDArray[np.uint8]
    action_ids: npt.NDArray[np.uint16]
    visit_counts: npt.NDArray[np.uint16]
    legal_count: npt.NDArray[np.uint8]
    legal_action_ids: npt.NDArray[np.uint16]


@dataclass(frozen=True)
class ReplayNextPolicyColumnViews:
    kind: Literal['next_policy']
    policy: ReplayPolicyColumnViews
    eligible: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ReplayScalarColumnViews:
    kind: Literal['remaining_game_length', 'future_search_value', 'irreversible_progress']
    value: npt.NDArray[np.float32]
    eligible: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ReplaySearchCorrectionColumnViews:
    kind: Literal['search_correction']
    value: npt.NDArray[np.float32]


@dataclass(frozen=True)
class ReplayLegalMovesColumnViews:
    kind: Literal['legal_moves']


ReplayAuxiliaryColumnViews: TypeAlias = (
    ReplayNextPolicyColumnViews
    | ReplayScalarColumnViews
    | ReplaySearchCorrectionColumnViews
    | ReplayLegalMovesColumnViews
)


@dataclass(frozen=True)
class ReplayColumnViews:
    encoded_state: npt.NDArray[np.uint8]
    policy: ReplayPolicyColumnViews
    wdl_target: npt.NDArray[np.float32]
    root_value: npt.NDArray[np.float32]
    auxiliary: tuple[ReplayAuxiliaryColumnViews, ...]
    sample_weight: npt.NDArray[np.float32]
    source_model_generation: npt.NDArray[np.uint32]
    source_timestamp: npt.NDArray[np.float64]

    @property
    def row_count(self) -> int:
        return len(self.encoded_state)


def build_column_views(
    layout: ReplayLayout,
    arrays: tuple[ReplayColumnArray, ...],
) -> ReplayColumnViews:
    policy = _policy_views(arrays, None)
    auxiliary: list[ReplayAuxiliaryColumnViews] = []
    for index, head in enumerate(layout.targets.auxiliary_heads):
        match head:
            case NextPolicyHeadLayout():
                auxiliary.append(
                    ReplayNextPolicyColumnViews(
                        kind='next_policy',
                        policy=_policy_views(arrays, index),
                        eligible=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_ELIGIBLE, index),
                    )
                )
            case RemainingGameLengthHeadLayout():
                auxiliary.append(
                    ReplayScalarColumnViews(
                        kind='remaining_game_length',
                        value=_float32_array(arrays, ReplayColumnKind.AUXILIARY_VALUE, index),
                        eligible=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_ELIGIBLE, index),
                    )
                )
            case FutureSearchValueHeadLayout():
                auxiliary.append(
                    ReplayScalarColumnViews(
                        kind='future_search_value',
                        value=_float32_array(arrays, ReplayColumnKind.AUXILIARY_VALUE, index),
                        eligible=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_ELIGIBLE, index),
                    )
                )
            case IrreversibleProgressHeadLayout():
                auxiliary.append(
                    ReplayScalarColumnViews(
                        kind='irreversible_progress',
                        value=_float32_array(arrays, ReplayColumnKind.AUXILIARY_VALUE, index),
                        eligible=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_ELIGIBLE, index),
                    )
                )
            case SearchCorrectionHeadLayout():
                auxiliary.append(
                    ReplaySearchCorrectionColumnViews(
                        kind='search_correction',
                        value=_float32_array(arrays, ReplayColumnKind.AUXILIARY_VALUE, index),
                    )
                )
            case LegalMovesHeadLayout():
                auxiliary.append(ReplayLegalMovesColumnViews(kind='legal_moves'))
    return ReplayColumnViews(
        encoded_state=_uint8_array(arrays, ReplayColumnKind.ENCODED_STATE),
        policy=policy,
        wdl_target=_float32_array(arrays, ReplayColumnKind.WDL_TARGET),
        root_value=_float32_array(arrays, ReplayColumnKind.ROOT_VALUE),
        auxiliary=tuple(auxiliary),
        sample_weight=_float32_array(arrays, ReplayColumnKind.SAMPLE_WEIGHT),
        source_model_generation=_uint32_array(arrays, ReplayColumnKind.SOURCE_MODEL_GENERATION),
        source_timestamp=_float64_array(arrays, ReplayColumnKind.SOURCE_TIMESTAMP),
    )


def flatten_column_views(
    layout: ReplayLayout,
    views: ReplayColumnViews,
) -> tuple[ReplayColumnArray, ...]:
    values: list[ReplayColumnArray] = []
    for descriptor in layout.columns.columns:
        values.append(ReplayColumnArray(descriptor, _values_for_descriptor(views, descriptor)))
    return tuple(values)


def _policy_views(
    arrays: tuple[ReplayColumnArray, ...],
    auxiliary_index: int | None,
) -> ReplayPolicyColumnViews:
    if auxiliary_index is None:
        return ReplayPolicyColumnViews(
            entry_count=_uint8_array(arrays, ReplayColumnKind.POLICY_ENTRY_COUNT),
            action_ids=_uint16_array(arrays, ReplayColumnKind.POLICY_ACTION_IDS),
            visit_counts=_uint16_array(arrays, ReplayColumnKind.POLICY_VISIT_COUNTS),
            legal_count=_uint8_array(arrays, ReplayColumnKind.POLICY_LEGAL_COUNT),
            legal_action_ids=_uint16_array(arrays, ReplayColumnKind.POLICY_LEGAL_ACTION_IDS),
        )
    return ReplayPolicyColumnViews(
        entry_count=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_ENTRY_COUNT, auxiliary_index),
        action_ids=_uint16_array(arrays, ReplayColumnKind.AUXILIARY_ACTION_IDS, auxiliary_index),
        visit_counts=_uint16_array(arrays, ReplayColumnKind.AUXILIARY_VISIT_COUNTS, auxiliary_index),
        legal_count=_uint8_array(arrays, ReplayColumnKind.AUXILIARY_LEGAL_COUNT, auxiliary_index),
        legal_action_ids=_uint16_array(arrays, ReplayColumnKind.AUXILIARY_LEGAL_ACTION_IDS, auxiliary_index),
    )


def _values_for_descriptor(
    views: ReplayColumnViews,
    descriptor: ReplayColumnDescriptor,
) -> ReplayArray:
    key = descriptor.key
    match key.kind:
        case ReplayColumnKind.ENCODED_STATE:
            return views.encoded_state
        case ReplayColumnKind.POLICY_ENTRY_COUNT:
            return views.policy.entry_count
        case ReplayColumnKind.POLICY_ACTION_IDS:
            return views.policy.action_ids
        case ReplayColumnKind.POLICY_VISIT_COUNTS:
            return views.policy.visit_counts
        case ReplayColumnKind.POLICY_LEGAL_COUNT:
            return views.policy.legal_count
        case ReplayColumnKind.POLICY_LEGAL_ACTION_IDS:
            return views.policy.legal_action_ids
        case ReplayColumnKind.WDL_TARGET:
            return views.wdl_target
        case ReplayColumnKind.ROOT_VALUE:
            return views.root_value
        case ReplayColumnKind.SAMPLE_WEIGHT:
            return views.sample_weight
        case ReplayColumnKind.SOURCE_MODEL_GENERATION:
            return views.source_model_generation
        case ReplayColumnKind.SOURCE_TIMESTAMP:
            return views.source_timestamp
        case _:
            pass
    auxiliary_index = key.auxiliary_index
    assert auxiliary_index is not None
    target = views.auxiliary[auxiliary_index]
    match target, key.kind:
        case ReplayNextPolicyColumnViews(policy=policy), ReplayColumnKind.AUXILIARY_ENTRY_COUNT:
            return policy.entry_count
        case ReplayNextPolicyColumnViews(policy=policy), ReplayColumnKind.AUXILIARY_ACTION_IDS:
            return policy.action_ids
        case ReplayNextPolicyColumnViews(policy=policy), ReplayColumnKind.AUXILIARY_VISIT_COUNTS:
            return policy.visit_counts
        case ReplayNextPolicyColumnViews(policy=policy), ReplayColumnKind.AUXILIARY_LEGAL_COUNT:
            return policy.legal_count
        case ReplayNextPolicyColumnViews(policy=policy), ReplayColumnKind.AUXILIARY_LEGAL_ACTION_IDS:
            return policy.legal_action_ids
        case ReplayNextPolicyColumnViews(eligible=eligible), ReplayColumnKind.AUXILIARY_ELIGIBLE:
            return eligible
        case ReplayScalarColumnViews(value=value), ReplayColumnKind.AUXILIARY_VALUE:
            return value
        case ReplayScalarColumnViews(eligible=eligible), ReplayColumnKind.AUXILIARY_ELIGIBLE:
            return eligible
        case ReplaySearchCorrectionColumnViews(value=value), ReplayColumnKind.AUXILIARY_VALUE:
            return value
        case _:
            raise ValueError(f'Replay column {key.name} does not match its auxiliary target.')


def _array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> ReplayArray:
    key = ReplayColumnKey(kind, auxiliary_index)
    return next(column.values for column in arrays if column.descriptor.key == key)


def _uint8_array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> npt.NDArray[np.uint8]:
    return cast(npt.NDArray[np.uint8], _array(arrays, kind, auxiliary_index))


def _uint16_array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> npt.NDArray[np.uint16]:
    return cast(npt.NDArray[np.uint16], _array(arrays, kind, auxiliary_index))


def _uint32_array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> npt.NDArray[np.uint32]:
    return cast(npt.NDArray[np.uint32], _array(arrays, kind, auxiliary_index))


def _float32_array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> npt.NDArray[np.float32]:
    return cast(npt.NDArray[np.float32], _array(arrays, kind, auxiliary_index))


def _float64_array(
    arrays: tuple[ReplayColumnArray, ...],
    kind: ReplayColumnKind,
    auxiliary_index: int | None = None,
) -> npt.NDArray[np.float64]:
    return cast(npt.NDArray[np.float64], _array(arrays, kind, auxiliary_index))
