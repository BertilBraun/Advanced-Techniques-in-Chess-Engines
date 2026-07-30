from __future__ import annotations

import hashlib
import struct

import numpy as np

from src.az.games.go.configuration import GO_PAYLOAD_SCHEMA_VERSION, GoGameConfiguration
from src.az.games.go.samples import (
    DensePolicyTarget,
    GoBatch,
    GoSample,
    SparsePolicyTarget,
    create_batch,
)
from src.az.games.api import GameIdentifier


PAYLOAD_MAGIC = b'AZGOPAY1'
CHECKSUM_SIZE = hashlib.sha256().digest_size
HEADER = struct.Struct('<8sHBHBBfffI')
POLICY_DENSE = 0
POLICY_SPARSE = 1
VALUE_PRESENT = 1


class GoReplayCodec:
    def __init__(self, configuration: GoGameConfiguration, payload_schema_version: int) -> None:
        if payload_schema_version != GO_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(f'Go replay supports payload schema {GO_PAYLOAD_SCHEMA_VERSION}.')
        self._configuration = configuration
        self._payload_schema_version = payload_schema_version

    @property
    def game_identifier(self) -> GameIdentifier:
        return GameIdentifier.GO

    @property
    def payload_schema_version(self) -> int:
        return self._payload_schema_version

    def encode(self, sample: GoSample) -> bytes:
        sample.validate_configuration(self._configuration)
        match sample.policy_target:
            case DensePolicyTarget(probabilities=probabilities):
                policy_kind = POLICY_DENSE
                policy_count = len(probabilities)
                policy_bytes = probabilities.astype('<f4', copy=False).tobytes(order='C')
            case SparsePolicyTarget(actions=actions, weights=weights):
                policy_kind = POLICY_SPARSE
                policy_count = len(actions)
                policy_bytes = actions.astype('<u2', copy=False).tobytes(order='C')
                policy_bytes += weights.astype('<f4', copy=False).tobytes(order='C')
        flags = VALUE_PRESENT if sample.value_target is not None else 0
        value_target = 0.0 if sample.value_target is None else sample.value_target
        header = HEADER.pack(
            PAYLOAD_MAGIC,
            self._payload_schema_version,
            self._configuration.board_size,
            self._configuration.history_length,
            policy_kind,
            flags,
            sample.policy_weight,
            value_target,
            sample.value_weight,
            policy_count,
        )
        plane_bytes = sample.input_planes.astype(np.uint8, copy=False).tobytes(order='C')
        legal_bytes = np.packbits(sample.legal_action_mask, bitorder='little').tobytes()
        body = header + plane_bytes + legal_bytes + policy_bytes
        return body + hashlib.sha256(body).digest()

    def decode(self, payload: bytes) -> GoSample:
        if len(payload) < HEADER.size + CHECKSUM_SIZE:
            raise ValueError('Go replay payload is truncated.')
        body = payload[:-CHECKSUM_SIZE]
        if hashlib.sha256(body).digest() != payload[-CHECKSUM_SIZE:]:
            raise ValueError('Go replay payload checksum mismatch.')
        (
            magic,
            schema_version,
            board_size,
            history_length,
            policy_kind,
            flags,
            policy_weight,
            value_target,
            value_weight,
            policy_count,
        ) = HEADER.unpack(body[: HEADER.size])
        if magic != PAYLOAD_MAGIC or schema_version != self._payload_schema_version:
            raise ValueError('Go replay payload has an unsupported identity or schema.')
        if board_size != self._configuration.board_size or history_length != self._configuration.history_length:
            raise ValueError('Go replay payload shape does not match the configured game.')
        action_count = self._configuration.action_count
        plane_value_count = self._configuration.input_plane_count * board_size**2
        plane_byte_count = plane_value_count
        legal_byte_count = (action_count + 7) // 8
        data_offset = HEADER.size
        policy_offset = data_offset + plane_byte_count + legal_byte_count
        planes = np.frombuffer(
            body[data_offset : data_offset + plane_byte_count],
            dtype=np.uint8,
        ).astype(np.float32)
        planes = planes.reshape(self._configuration.input_plane_count, board_size, board_size)
        legal = np.unpackbits(
            np.frombuffer(body[data_offset + plane_byte_count : policy_offset], dtype=np.uint8),
            bitorder='little',
        )[:action_count].astype(np.bool_)
        policy_data = body[policy_offset:]
        if policy_kind == POLICY_DENSE:
            expected_bytes = action_count * np.dtype('<f4').itemsize
            if policy_count != action_count or len(policy_data) != expected_bytes:
                raise ValueError('Dense Go policy payload has an invalid length.')
            policy = DensePolicyTarget(np.frombuffer(policy_data, dtype='<f4'))
        elif policy_kind == POLICY_SPARSE:
            expected_bytes = policy_count * (np.dtype('<u2').itemsize + np.dtype('<f4').itemsize)
            if policy_count == 0 or len(policy_data) != expected_bytes:
                raise ValueError('Sparse Go policy payload has an invalid length.')
            actions_size = policy_count * np.dtype('<u2').itemsize
            actions = np.frombuffer(policy_data[:actions_size], dtype='<u2').astype(np.int32)
            weights = np.frombuffer(policy_data[actions_size:], dtype='<f4')
            policy = SparsePolicyTarget(actions, weights)
        else:
            raise ValueError('Go replay payload has an unknown policy representation.')
        if flags not in (0, VALUE_PRESENT):
            raise ValueError('Go replay payload contains unknown flags.')
        return GoSample(
            input_planes=planes,
            legal_action_mask=legal,
            policy_target=policy,
            policy_weight=policy_weight,
            value_target=value_target if flags & VALUE_PRESENT else None,
            value_weight=value_weight,
        )

    def create_batch(self, samples: tuple[GoSample, ...]) -> GoBatch:
        return create_batch(samples, self._configuration)
