from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from pydantic import Field, model_validator
from src.games.representation import PackedPlaneLayout
from src.training.targets import (
    AuxiliaryHeadLayout,
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    TrainingTargetLayout,
)
from src.util.frozen_model import FrozenModel

ReplayDtypeField = tuple[str, str] | tuple[str, str, tuple[int, ...]]
ReplayNumpyDtype: TypeAlias = (
    np.dtype[np.uint8] | np.dtype[np.uint16] | np.dtype[np.uint32] | np.dtype[np.float32] | np.dtype[np.float64]
)


class ReplayElementType(str, Enum):
    UINT8 = 'u1'
    UINT16 = '<u2'
    UINT32 = '<u4'
    FLOAT32 = '<f4'
    FLOAT64 = '<f8'

    @property
    def numpy_dtype(self) -> ReplayNumpyDtype:
        match self:
            case ReplayElementType.UINT8:
                return np.dtype(np.uint8)
            case ReplayElementType.UINT16:
                return np.dtype('<u2')
            case ReplayElementType.UINT32:
                return np.dtype('<u4')
            case ReplayElementType.FLOAT32:
                return np.dtype('<f4')
            case ReplayElementType.FLOAT64:
                return np.dtype('<f8')


class ReplayColumnKind(str, Enum):
    ENCODED_STATE = 'encoded_state'
    POLICY_ENTRY_COUNT = 'policy_entry_count'
    POLICY_ACTION_IDS = 'policy_action_ids'
    POLICY_VISIT_COUNTS = 'policy_visit_counts'
    POLICY_LEGAL_COUNT = 'policy_legal_count'
    POLICY_LEGAL_ACTION_IDS = 'policy_legal_action_ids'
    WDL_TARGET = 'wdl_target'
    ROOT_VALUE = 'root_value'
    AUXILIARY_ENTRY_COUNT = 'auxiliary_entry_count'
    AUXILIARY_ACTION_IDS = 'auxiliary_action_ids'
    AUXILIARY_VISIT_COUNTS = 'auxiliary_visit_counts'
    AUXILIARY_LEGAL_COUNT = 'auxiliary_legal_count'
    AUXILIARY_LEGAL_ACTION_IDS = 'auxiliary_legal_action_ids'
    AUXILIARY_VALUE = 'auxiliary_value'
    AUXILIARY_ELIGIBLE = 'auxiliary_eligible'
    SAMPLE_WEIGHT = 'sample_weight'
    SOURCE_MODEL_GENERATION = 'source_model_generation'
    SOURCE_TIMESTAMP = 'source_timestamp'


@dataclass(frozen=True)
class ReplayColumnKey:
    kind: ReplayColumnKind
    auxiliary_index: int | None = None

    def __post_init__(self) -> None:
        is_auxiliary = self.kind.value.startswith('auxiliary_')
        if is_auxiliary != (self.auxiliary_index is not None):
            raise ValueError('Replay auxiliary column keys require exactly one auxiliary index.')
        if self.auxiliary_index is not None and self.auxiliary_index < 0:
            raise ValueError('Replay auxiliary column indices must be nonnegative.')

    @property
    def name(self) -> str:
        if self.auxiliary_index is None:
            return self.kind.value
        suffix = self.kind.value.removeprefix('auxiliary_')
        return f'auxiliary_{self.auxiliary_index}_{suffix}'


@dataclass(frozen=True)
class ReplayColumnDescriptor:
    key: ReplayColumnKey
    element_type: ReplayElementType
    trailing_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.key.name.encode('ascii')) > 64:
            raise ValueError('Replay column names must fit the binary descriptor.')
        if len(self.element_type.value.encode('ascii')) > 8:
            raise ValueError('Replay column dtypes must fit the binary descriptor.')
        if len(self.trailing_shape) > 4:
            raise ValueError('Replay columns support at most four trailing dimensions.')
        if any(dimension <= 0 for dimension in self.trailing_shape):
            raise ValueError('Replay column dimensions must be positive.')

    @property
    def row_bytes(self) -> int:
        element_count = int(np.prod(self.trailing_shape, dtype=np.int64)) if self.trailing_shape else 1
        return element_count * self.element_type.numpy_dtype.itemsize


@dataclass(frozen=True)
class ReplayColumnLayout:
    columns: tuple[ReplayColumnDescriptor, ...]

    def __post_init__(self) -> None:
        keys = tuple(column.key for column in self.columns)
        if len(set(keys)) != len(keys):
            raise ValueError('Replay column keys must be unique.')


class ReplayLayout(FrozenModel):
    packed_planes: PackedPlaneLayout
    targets: TrainingTargetLayout
    maximum_policy_entries: int = Field(ge=1, le=255)
    maximum_legal_actions: int = Field(ge=1, le=255)

    @model_validator(mode='after')
    def validate_dimensions(self) -> ReplayLayout:
        if self.targets.action_size > 65_536:
            raise ValueError('Replay action IDs must fit uint16.')
        if self.maximum_policy_entries > self.targets.action_size:
            raise ValueError('Maximum retained policy entries cannot exceed the action count.')
        if self.maximum_legal_actions > self.targets.action_size:
            raise ValueError('Maximum legal actions cannot exceed the action count.')
        return self

    @property
    def columns(self) -> ReplayColumnLayout:
        descriptors = [
            ReplayColumnDescriptor(
                ReplayColumnKey(ReplayColumnKind.ENCODED_STATE),
                ReplayElementType.UINT8,
                (self.packed_planes.payload_bytes,),
            ),
            ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.POLICY_ENTRY_COUNT), ReplayElementType.UINT8),
            ReplayColumnDescriptor(
                ReplayColumnKey(ReplayColumnKind.POLICY_ACTION_IDS),
                ReplayElementType.UINT16,
                (self.maximum_policy_entries,),
            ),
            ReplayColumnDescriptor(
                ReplayColumnKey(ReplayColumnKind.POLICY_VISIT_COUNTS),
                ReplayElementType.UINT16,
                (self.maximum_policy_entries,),
            ),
            ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.POLICY_LEGAL_COUNT), ReplayElementType.UINT8),
            ReplayColumnDescriptor(
                ReplayColumnKey(ReplayColumnKind.POLICY_LEGAL_ACTION_IDS),
                ReplayElementType.UINT16,
                (self.maximum_legal_actions,),
            ),
            ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.WDL_TARGET), ReplayElementType.FLOAT32, (3,)),
            ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.ROOT_VALUE), ReplayElementType.FLOAT32),
        ]
        for index, head in enumerate(self.targets.auxiliary_heads):
            match head:
                case NextPolicyHeadLayout():
                    descriptors.extend(
                        (
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_ENTRY_COUNT, index),
                                ReplayElementType.UINT8,
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_ACTION_IDS, index),
                                ReplayElementType.UINT16,
                                (self.maximum_policy_entries,),
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_VISIT_COUNTS, index),
                                ReplayElementType.UINT16,
                                (self.maximum_policy_entries,),
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_LEGAL_COUNT, index),
                                ReplayElementType.UINT8,
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_LEGAL_ACTION_IDS, index),
                                ReplayElementType.UINT16,
                                (self.maximum_legal_actions,),
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_ELIGIBLE, index),
                                ReplayElementType.UINT8,
                            ),
                        )
                    )
                case RemainingGameLengthHeadLayout() | FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout():
                    descriptors.extend(
                        (
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_VALUE, index), ReplayElementType.FLOAT32
                            ),
                            ReplayColumnDescriptor(
                                ReplayColumnKey(ReplayColumnKind.AUXILIARY_ELIGIBLE, index), ReplayElementType.UINT8
                            ),
                        )
                    )
                case LegalMovesHeadLayout():
                    # Legal-moves targets are derived from the primary legal-action columns.
                    pass
        descriptors.extend(
            (
                ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.SAMPLE_WEIGHT), ReplayElementType.FLOAT32),
                ReplayColumnDescriptor(
                    ReplayColumnKey(ReplayColumnKind.SOURCE_MODEL_GENERATION), ReplayElementType.UINT32
                ),
                ReplayColumnDescriptor(ReplayColumnKey(ReplayColumnKind.SOURCE_TIMESTAMP), ReplayElementType.FLOAT64),
            )
        )
        return ReplayColumnLayout(tuple(descriptors))

    @property
    def row_dtype(self) -> np.dtype[np.void]:
        fields: list[ReplayDtypeField] = []
        for descriptor in self.columns.columns:
            if descriptor.trailing_shape:
                fields.append((descriptor.key.name, descriptor.element_type.value, descriptor.trailing_shape))
            else:
                fields.append((descriptor.key.name, descriptor.element_type.value))
        return np.dtype(fields, align=False)

    @property
    def row_bytes(self) -> int:
        return self.row_dtype.itemsize

    @property
    def digest(self) -> str:
        payload = {
            'schema_version': 5,
            'packed_planes': {
                'board_size': self.packed_planes.board_size,
                'binary_plane_count': self.packed_planes.binary_plane_count,
                'scalar_count': self.packed_planes.scalar_count,
                'payload_bytes': self.packed_planes.payload_bytes,
            },
            'targets': {
                'action_size': self.targets.action_size,
                'wdl_size': self.targets.wdl_size,
                'auxiliary_heads': [_head_digest_fields(head) for head in self.targets.auxiliary_heads],
            },
            'maximum_policy_entries': self.maximum_policy_entries,
            'maximum_legal_actions': self.maximum_legal_actions,
            'columns': [
                {
                    'kind': descriptor.key.kind.value,
                    'auxiliary_index': descriptor.key.auxiliary_index,
                    'dtype': descriptor.element_type.value,
                    'trailing_shape': descriptor.trailing_shape,
                }
                for descriptor in self.columns.columns
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()


def _head_digest_fields(head: AuxiliaryHeadLayout) -> dict[str, int | float | str]:
    match head:
        case NextPolicyHeadLayout(action_size=action_size, ply_offset=ply_offset):
            return {'kind': head.kind, 'action_size': action_size, 'ply_offset': ply_offset}
        case RemainingGameLengthHeadLayout(normalization_scale=normalization_scale):
            return {'kind': head.kind, 'output_size': 1, 'normalization_scale': normalization_scale}
        case FutureSearchValueHeadLayout(ply_offset=ply_offset, smooth_l1_beta=beta):
            return {'kind': head.kind, 'output_size': 1, 'ply_offset': ply_offset, 'smooth_l1_beta': beta}
        case IrreversibleProgressHeadLayout(horizon_plies=horizon_plies):
            return {'kind': head.kind, 'output_size': 1, 'horizon_plies': horizon_plies}
        case LegalMovesHeadLayout(action_size=action_size):
            return {'kind': head.kind, 'action_size': action_size}
