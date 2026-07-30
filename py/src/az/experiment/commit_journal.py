from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from pydantic import model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.serialization import model_sha256
from src.az.replay.envelope import ReplayEnvelope, ReplayRecord
from src.az.runtime.telemetry_journal import TelemetryJournal


class CommittedReplayEnvelope(FrozenModel):
    envelope: ReplayEnvelope
    envelope_sha256: Sha256

    @model_validator(mode='after')
    def validate_digest(self) -> CommittedReplayEnvelope:
        if self.envelope_sha256 != model_sha256(self.envelope):
            raise ValueError('Committed replay envelope digest mismatch.')
        return self


class ReplayCommitJournal:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ValueError('Replay commit journal path must be absolute.')
        self._path = path
        self._journal = TelemetryJournal(path)
        entries = tuple(
            CommittedReplayEnvelope.model_validate_json(payload) for payload in self._journal.read_payloads()
        )
        self._envelopes = {entry.envelope.sample_id: entry for entry in entries}
        if len(self._envelopes) != len(entries):
            raise ValueError('Replay commit journal contains duplicate sample identities.')

    @property
    def path(self) -> Path:
        return self._path

    @property
    def envelopes(self) -> tuple[ReplayEnvelope, ...]:
        return tuple(entry.envelope for entry in self._envelopes.values())

    @property
    def sample_ids(self) -> frozenset[UUID]:
        return frozenset(self._envelopes)

    @property
    def prefix_sha256(self) -> Sha256:
        digest = hashlib.sha256()
        for entry in self._envelopes.values():
            digest.update(bytes.fromhex(entry.envelope_sha256))
        return digest.hexdigest()

    def next_game_indices(self, logical_worker_count: int) -> tuple[int, ...]:
        if logical_worker_count <= 0:
            raise ValueError('Logical worker count must be positive.')
        next_indices = [0] * logical_worker_count
        for entry in self._envelopes.values():
            lineage = entry.envelope.seed_lineage
            if lineage.worker_index >= logical_worker_count:
                raise ValueError('Committed replay worker is outside the configured topology.')
            next_indices[lineage.worker_index] = max(
                next_indices[lineage.worker_index],
                lineage.game_index + 1,
            )
        return tuple(next_indices)

    def commit(self, records: tuple[ReplayRecord, ...]) -> None:
        additions: list[CommittedReplayEnvelope] = []
        for record in records:
            entry = CommittedReplayEnvelope(
                envelope=record.envelope,
                envelope_sha256=model_sha256(record.envelope),
            )
            existing = self._envelopes.get(record.envelope.sample_id)
            if existing is not None:
                if existing != entry:
                    raise ValueError('Replay sample identity has conflicting committed evidence.')
                continue
            additions.append(entry)
        if not additions:
            return
        self._journal.append(tuple(entry.model_dump_json().encode() for entry in additions))
        self._envelopes.update((entry.envelope.sample_id, entry) for entry in additions)
