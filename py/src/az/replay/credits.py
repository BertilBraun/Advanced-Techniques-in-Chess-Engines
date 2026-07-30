from __future__ import annotations

import hashlib
import os
import struct
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256


CREDIT_JOURNAL_MAGIC = b'AZCRDT02'
CREDIT_FRAME = struct.Struct('<II')
CREDIT_SHARD_HEADER = struct.Struct('<QI')
CHECKSUM_SIZE = hashlib.sha256().digest_size
EMPTY_CREDIT_PREFIX_SHA256 = hashlib.sha256(b'').hexdigest()
UINT32_MAXIMUM = 2**32 - 1


class ReplayCreditSnapshot(FrozenModel):
    credited_unique_positions: int = Field(ge=0)
    prefix_sha256: Sha256


class ReplayCreditState(FrozenModel):
    credited_unique_positions: int = Field(ge=0)
    credit_journal_prefix_sha256: Sha256
    earned_position_credits: Decimal = Field(ge=0)
    consumed_position_credits: Decimal = Field(ge=0)
    available_position_credits: Decimal = Field(ge=0)
    completed_optimizer_steps: int = Field(ge=0)
    completed_training_quanta: int = Field(ge=0)
    model_version: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_accounting(self) -> ReplayCreditState:
        if self.credited_unique_positions == 0 and self.credit_journal_prefix_sha256 != EMPTY_CREDIT_PREFIX_SHA256:
            raise ValueError('An empty replay credit state must use the empty journal prefix digest.')
        if self.available_position_credits != self.earned_position_credits - self.consumed_position_credits:
            raise ValueError('Available replay credits must equal earned credits minus consumed credits.')
        if self.consumed_position_credits > self.earned_position_credits:
            raise ValueError('Consumed replay credits cannot exceed earned credits.')
        if self.model_version != self.completed_training_quanta:
            raise ValueError('Model version must equal the number of published training quanta.')
        return self

    @classmethod
    def initial(cls) -> ReplayCreditState:
        return cls(
            credited_unique_positions=0,
            credit_journal_prefix_sha256=EMPTY_CREDIT_PREFIX_SHA256,
            earned_position_credits=Decimal(0),
            consumed_position_credits=Decimal(0),
            available_position_credits=Decimal(0),
            completed_optimizer_steps=0,
            completed_training_quanta=0,
            model_version=0,
        )

    def reconcile(
        self,
        snapshot: ReplayCreditSnapshot,
        target_reuse: Decimal,
    ) -> ReplayCreditState:
        if not target_reuse.is_finite() or target_reuse <= 0:
            raise ValueError('Replay target reuse must be finite and positive.')
        if snapshot.credited_unique_positions < self.credited_unique_positions:
            raise ValueError('Durably credited replay positions cannot move backwards.')
        earned = Decimal(snapshot.credited_unique_positions) * target_reuse
        if earned < self.consumed_position_credits:
            raise ValueError('Durable replay credits do not cover already published training.')
        return ReplayCreditState(
            credited_unique_positions=snapshot.credited_unique_positions,
            credit_journal_prefix_sha256=snapshot.prefix_sha256,
            earned_position_credits=earned,
            consumed_position_credits=self.consumed_position_credits,
            available_position_credits=earned - self.consumed_position_credits,
            completed_optimizer_steps=self.completed_optimizer_steps,
            completed_training_quanta=self.completed_training_quanta,
            model_version=self.model_version,
        )

    def prepare_training_quantum(
        self,
        optimizer_steps: int,
        global_batch_size: int,
        maximum_optimizer_steps: int,
    ) -> ReplayCreditState:
        if optimizer_steps <= 0 or global_batch_size <= 0:
            raise ValueError('Training quantum steps and global batch size must be positive.')
        next_optimizer_steps = self.completed_optimizer_steps + optimizer_steps
        if next_optimizer_steps > maximum_optimizer_steps:
            raise ValueError('Training quantum exceeds the optimizer-step limit.')
        required = Decimal(optimizer_steps * global_batch_size)
        if self.available_position_credits < required:
            raise ValueError('Insufficient replay credits for a complete training quantum.')
        consumed = self.consumed_position_credits + required
        return ReplayCreditState(
            credited_unique_positions=self.credited_unique_positions,
            credit_journal_prefix_sha256=self.credit_journal_prefix_sha256,
            earned_position_credits=self.earned_position_credits,
            consumed_position_credits=consumed,
            available_position_credits=self.earned_position_credits - consumed,
            completed_optimizer_steps=next_optimizer_steps,
            completed_training_quanta=self.completed_training_quanta + 1,
            model_version=self.model_version + 1,
        )


class ReplayCreditJournal:
    """Append-only shard-aware identity ledger for exact replay credit accounting."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._shards, self._ordered_identities, self._prefix_digests = self._load()
        self._identities = set(self._ordered_identities)

    @property
    def credited_unique_positions(self) -> int:
        return len(self._identities)

    @property
    def snapshot(self) -> ReplayCreditSnapshot:
        return ReplayCreditSnapshot(
            credited_unique_positions=len(self._ordered_identities),
            prefix_sha256=self._prefix_digests[-1],
        )

    def verify_snapshot(self, snapshot: ReplayCreditSnapshot) -> None:
        if snapshot.credited_unique_positions > len(self._ordered_identities):
            raise ValueError('Replay credit journal is behind the checkpoint snapshot.')
        if self._prefix_digests[snapshot.credited_unique_positions] != snapshot.prefix_sha256:
            raise ValueError('Replay credit journal prefix does not match the checkpoint snapshot.')

    @property
    def latest_shard_sequence(self) -> int | None:
        return max(self._shards) if self._shards else None

    def has_shard(self, sequence: int) -> bool:
        return sequence in self._shards

    def preflight_new_shard(self, sequence: int, credit_ids: tuple[UUID, ...]) -> None:
        self._validate_shard(sequence, credit_ids)
        if sequence in self._shards:
            raise ValueError('Replay shard sequence already exists in the credit journal.')
        latest = self.latest_shard_sequence
        if latest is not None and sequence <= latest:
            raise ValueError('Replay shard sequence must increase beyond durable credit history.')
        if any(credit_id in self._identities for credit_id in credit_ids):
            raise ValueError('A newly published replay shard reused an existing credit identity.')

    def credit_shard(self, sequence: int, credit_ids: tuple[UUID, ...]) -> int:
        self._validate_shard(sequence, credit_ids)
        existing = self._shards.get(sequence)
        if existing is not None:
            if existing != credit_ids:
                raise ValueError('Replay shard credit identity set conflicts with durable journal history.')
            return len(self._identities)
        latest = self.latest_shard_sequence
        if latest is not None and sequence <= latest:
            raise ValueError('Replay shard sequence conflicts with durable credit history.')
        if any(credit_id in self._identities for credit_id in credit_ids):
            raise ValueError('Replay shard reused a credit identity from another shard.')
        payload = CREDIT_SHARD_HEADER.pack(sequence, len(credit_ids))
        payload += b''.join(credit_id.bytes for credit_id in credit_ids)
        if len(payload) > UINT32_MAXIMUM:
            raise ValueError('Replay shard credit record exceeds uint32 framing.')
        frame = CREDIT_FRAME.pack(len(payload), len(payload) ^ UINT32_MAXIMUM)
        encoded = frame + payload
        checksum = hashlib.sha256(encoded).digest()
        with self._path.open('ab') as stream:
            stream.write(encoded)
            stream.write(checksum)
            stream.flush()
            os.fsync(stream.fileno())
        self._shards[sequence] = credit_ids
        self._identities.update(credit_ids)
        prefix_digest = hashlib.sha256(
            bytes.fromhex(self._prefix_digests[-1]) + CREDIT_SHARD_HEADER.pack(sequence, len(credit_ids))
        ).hexdigest()
        for credit_id in credit_ids:
            self._ordered_identities.append(credit_id)
            prefix_digest = hashlib.sha256(bytes.fromhex(prefix_digest) + credit_id.bytes).hexdigest()
            self._prefix_digests.append(prefix_digest)
        return len(self._identities)

    @staticmethod
    def _validate_shard(sequence: int, credit_ids: tuple[UUID, ...]) -> None:
        if not 0 <= sequence <= 2**64 - 1:
            raise ValueError('Replay shard credit sequence must fit uint64.')
        if not credit_ids:
            raise ValueError('Replay shard credit identity set cannot be empty.')
        if len(credit_ids) > UINT32_MAXIMUM:
            raise ValueError('Replay shard credit identity count must fit uint32.')
        if len(set(credit_ids)) != len(credit_ids):
            raise ValueError('Replay shard contains duplicate credit identities.')

    def _load(self) -> tuple[dict[int, tuple[UUID, ...]], list[UUID], list[str]]:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open('xb') as stream:
                stream.write(CREDIT_JOURNAL_MAGIC)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(self._path.parent)
        contents = self._path.read_bytes()
        if not contents.startswith(CREDIT_JOURNAL_MAGIC):
            raise ValueError('Replay credit journal has an invalid header.')
        body = contents[len(CREDIT_JOURNAL_MAGIC) :]
        shards: dict[int, tuple[UUID, ...]] = {}
        identities: list[UUID] = []
        identity_set: set[UUID] = set()
        prefix_digests = [EMPTY_CREDIT_PREFIX_SHA256]
        offset = 0
        while offset < len(body):
            remaining = len(body) - offset
            if remaining < CREDIT_FRAME.size:
                self._truncate_journal(offset)
                break
            payload_size, complement = CREDIT_FRAME.unpack(body[offset : offset + CREDIT_FRAME.size])
            if payload_size ^ complement != UINT32_MAXIMUM:
                raise ValueError('Replay credit journal has corrupt record framing.')
            if payload_size < CREDIT_SHARD_HEADER.size or (payload_size - CREDIT_SHARD_HEADER.size) % 16:
                raise ValueError('Replay credit journal has invalid shard-credit framing.')
            record_size = CREDIT_FRAME.size + payload_size + CHECKSUM_SIZE
            if remaining < record_size:
                self._truncate_journal(offset)
                break
            encoded = body[offset : offset + CREDIT_FRAME.size + payload_size]
            checksum = body[offset + CREDIT_FRAME.size + payload_size : offset + record_size]
            if hashlib.sha256(encoded).digest() != checksum:
                raise ValueError('Replay credit journal record checksum mismatch.')
            payload = encoded[CREDIT_FRAME.size :]
            sequence, identity_count = CREDIT_SHARD_HEADER.unpack(payload[: CREDIT_SHARD_HEADER.size])
            identity_bytes = payload[CREDIT_SHARD_HEADER.size :]
            if len(identity_bytes) != identity_count * 16:
                raise ValueError('Replay credit journal shard count does not match its framing.')
            credit_ids = tuple(
                UUID(bytes=identity_bytes[index * 16 : (index + 1) * 16]) for index in range(identity_count)
            )
            self._validate_shard(sequence, credit_ids)
            if sequence in shards:
                raise ValueError('Replay credit journal contains a duplicate shard sequence.')
            if any(credit_id in identity_set for credit_id in credit_ids):
                raise ValueError('Replay credit journal contains a duplicate identity.')
            if shards and sequence <= max(shards):
                raise ValueError('Replay credit journal shard sequences are not strictly increasing.')
            shards[sequence] = credit_ids
            prefix_digest = hashlib.sha256(
                bytes.fromhex(prefix_digests[-1]) + CREDIT_SHARD_HEADER.pack(sequence, len(credit_ids))
            ).hexdigest()
            for credit_id in credit_ids:
                identities.append(credit_id)
                identity_set.add(credit_id)
                prefix_digest = hashlib.sha256(bytes.fromhex(prefix_digest) + credit_id.bytes).hexdigest()
                prefix_digests.append(prefix_digest)
            offset += record_size
        return shards, identities, prefix_digests

    def _truncate_journal(self, valid_body_size: int) -> None:
        with self._path.open('r+b') as stream:
            stream.truncate(len(CREDIT_JOURNAL_MAGIC) + valid_body_size)
            stream.flush()
            os.fsync(stream.fileno())


def _sync_directory(directory: Path) -> None:
    if os.name == 'nt':
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
