from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import BinaryIO


TELEMETRY_MAGIC = b'AZTELEM1'
FRAME_LENGTH = struct.Struct('<I')
CHECKSUM_SIZE = hashlib.sha256().digest_size


class TelemetryJournal:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError('Telemetry journal path must be absolute.')
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(TELEMETRY_MAGIC)
                stream.flush()
                os.fsync(stream.fileno())
        self._repair_torn_tail()

    def append(self, payloads: tuple[bytes, ...]) -> None:
        if not payloads:
            return
        with self._path.open('ab') as stream:
            for payload in payloads:
                if not payload:
                    raise ValueError('Telemetry payload cannot be empty.')
                stream.write(FRAME_LENGTH.pack(len(payload)))
                stream.write(payload)
                stream.write(hashlib.sha256(payload).digest())
            stream.flush()
            os.fsync(stream.fileno())

    def read_payloads(self) -> tuple[bytes, ...]:
        self._repair_torn_tail()
        with self._path.open('rb') as stream:
            if stream.read(len(TELEMETRY_MAGIC)) != TELEMETRY_MAGIC:
                raise ValueError('Telemetry journal header is invalid.')
            payloads: list[bytes] = []
            while length_bytes := stream.read(FRAME_LENGTH.size):
                if len(length_bytes) != FRAME_LENGTH.size:
                    raise AssertionError('Telemetry repair left a partial frame length.')
                (payload_size,) = FRAME_LENGTH.unpack(length_bytes)
                payload = stream.read(payload_size)
                checksum = stream.read(CHECKSUM_SIZE)
                if len(payload) != payload_size or len(checksum) != CHECKSUM_SIZE:
                    raise AssertionError('Telemetry repair left a partial frame.')
                if hashlib.sha256(payload).digest() != checksum:
                    raise ValueError('Telemetry journal frame checksum mismatch.')
                payloads.append(payload)
        return tuple(payloads)

    def _repair_torn_tail(self) -> None:
        with self._path.open('r+b') as stream:
            if stream.read(len(TELEMETRY_MAGIC)) != TELEMETRY_MAGIC:
                raise ValueError('Telemetry journal header is invalid.')
            valid_end = len(TELEMETRY_MAGIC)
            while True:
                length_bytes = stream.read(FRAME_LENGTH.size)
                if not length_bytes:
                    return
                if len(length_bytes) != FRAME_LENGTH.size:
                    self._truncate(stream, valid_end)
                    return
                (payload_size,) = FRAME_LENGTH.unpack(length_bytes)
                payload = stream.read(payload_size)
                checksum = stream.read(CHECKSUM_SIZE)
                if len(payload) != payload_size or len(checksum) != CHECKSUM_SIZE:
                    self._truncate(stream, valid_end)
                    return
                if hashlib.sha256(payload).digest() != checksum:
                    raise ValueError('Telemetry journal frame checksum mismatch.')
                valid_end = stream.tell()

    @staticmethod
    def _truncate(stream: BinaryIO, valid_end: int) -> None:
        stream.truncate(valid_end)
        stream.flush()
        os.fsync(stream.fileno())
