from __future__ import annotations

import hashlib
import json

import numpy as np
from pydantic import Field, model_validator

from src.games.representation import PackedPlaneLayout
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout, TrainingTargetLayout
from src.util.frozen_model import FrozenModel

ReplayDtypeField = tuple[str, str] | tuple[str, str, tuple[int, ...]]


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
    def row_dtype(self) -> np.dtype[np.void]:
        fields: list[ReplayDtypeField] = [
            ('encoded_state', 'u1', (self.packed_planes.payload_bytes,)),
            ('policy_entry_count', 'u1'),
            ('policy_action_ids', '<u2', (self.maximum_policy_entries,)),
            ('policy_visit_counts', '<u2', (self.maximum_policy_entries,)),
            ('policy_legal_count', 'u1'),
            ('policy_legal_action_ids', '<u2', (self.maximum_legal_actions,)),
            ('wdl_target', '<f4', (3,)),
            ('root_value', '<f4'),
        ]
        for index, head in enumerate(self.targets.auxiliary_heads):
            match head:
                case NextPolicyHeadLayout():
                    fields.extend(
                        (
                            (f'auxiliary_{index}_entry_count', 'u1'),
                            (f'auxiliary_{index}_action_ids', '<u2', (self.maximum_policy_entries,)),
                            (f'auxiliary_{index}_visit_counts', '<u2', (self.maximum_policy_entries,)),
                            (f'auxiliary_{index}_legal_count', 'u1'),
                            (f'auxiliary_{index}_legal_action_ids', '<u2', (self.maximum_legal_actions,)),
                            (f'auxiliary_{index}_eligible', 'u1'),
                        )
                    )
                case RemainingGameLengthHeadLayout():
                    fields.extend(
                        (
                            (f'auxiliary_{index}_value', '<f4'),
                            (f'auxiliary_{index}_eligible', 'u1'),
                        )
                    )
        fields.extend(
            (
                ('sample_weight', '<f4'),
                ('source_model_generation', '<u4'),
                ('source_timestamp', '<f8'),
            )
        )
        return np.dtype(fields, align=False)

    @property
    def row_bytes(self) -> int:
        return self.row_dtype.itemsize

    @property
    def digest(self) -> str:
        payload = {
            'packed_planes': {
                'board_size': self.packed_planes.board_size,
                'binary_plane_count': self.packed_planes.binary_plane_count,
                'scalar_count': self.packed_planes.scalar_count,
            },
            'targets': {
                'action_size': self.targets.action_size,
                'wdl_size': self.targets.wdl_size,
                'auxiliary_heads': [_head_digest_fields(head) for head in self.targets.auxiliary_heads],
            },
            'maximum_policy_entries': self.maximum_policy_entries,
            'maximum_legal_actions': self.maximum_legal_actions,
            'dtype': self.row_dtype.descr,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()


def _head_digest_fields(head: NextPolicyHeadLayout | RemainingGameLengthHeadLayout) -> dict[str, int | float | str]:
    match head:
        case NextPolicyHeadLayout(action_size=action_size, ply_offset=ply_offset):
            return {'kind': head.kind, 'action_size': action_size, 'ply_offset': ply_offset}
        case RemainingGameLengthHeadLayout(normalization_scale=normalization_scale):
            return {'kind': head.kind, 'output_size': 1, 'normalization_scale': normalization_scale}
