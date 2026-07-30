from __future__ import annotations

from typing import Protocol, TypeVar

from src.az.games.api import GameIdentifier


SampleType = TypeVar('SampleType')
BatchType = TypeVar('BatchType')


class ReplayCodec(Protocol[SampleType, BatchType]):
    @property
    def game_identifier(self) -> GameIdentifier: ...

    @property
    def payload_schema_version(self) -> int: ...

    def encode(self, sample: SampleType) -> bytes: ...

    def decode(self, payload: bytes) -> SampleType: ...

    def create_batch(self, samples: tuple[SampleType, ...]) -> BatchType: ...
