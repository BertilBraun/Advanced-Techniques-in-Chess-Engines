from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from src.games.contracts import WdlTarget
from src.replay.contracts import (
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    IneligibleNextPolicyTarget,
    IneligibleRemainingGameLengthTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.layout import ReplayLayout
from AlphaZeroCpp import GameSearchVisit
from src.training.targets import NextPolicyHeadLayout, RemainingGameLengthHeadLayout


_REPLAY_MAGIC = b'AZRPLY01'
_REPLAY_SCHEMA_VERSION = 1
_HEADER_DTYPE = np.dtype(
    [
        ('magic', 'S8'),
        ('schema_version', '<u2'),
        ('layout_digest', 'S64'),
        ('row_bytes', '<u4'),
        ('maximum_capacity', '<u8'),
        ('logical_capacity', '<u8'),
        ('head', '<u8'),
        ('size', '<u8'),
        ('evicted_rows', '<u8'),
    ],
    align=False,
)


@dataclass(frozen=True)
class ReplayStoreState:
    maximum_capacity: int
    logical_capacity: int
    head: int
    size: int
    evicted_rows: int


class ReplayStore:
    def __init__(
        self,
        path: Path,
        layout: ReplayLayout,
        file: BinaryIO,
        mapping: mmap.mmap,
        header: npt.NDArray[np.void],
        rows: npt.NDArray[np.void],
        writable: bool,
    ) -> None:
        self.path = path
        self.layout = layout
        self._file = file
        self._mapping = mapping
        self._header = header
        self._rows = rows
        self._writable = writable
        self._closed = False

    @classmethod
    def projected_file_size(cls, layout: ReplayLayout, maximum_capacity: int) -> int:
        if maximum_capacity <= 0:
            raise ValueError('Replay maximum capacity must be positive.')
        return _HEADER_DTYPE.itemsize + maximum_capacity * layout.row_bytes

    @classmethod
    def create(
        cls,
        path: Path,
        layout: ReplayLayout,
        maximum_capacity: int,
        logical_capacity: int,
    ) -> ReplayStore:
        cls._validate_capacities(maximum_capacity, logical_capacity)
        if path.exists():
            raise ValueError(f'Replay store already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        file_size = cls.projected_file_size(layout, maximum_capacity)
        with path.open('wb') as new_file:
            new_file.truncate(file_size)
        store = cls._map(path, layout, writable=True, maximum_capacity=maximum_capacity)
        store._header[0]['magic'] = _REPLAY_MAGIC
        store._header[0]['schema_version'] = _REPLAY_SCHEMA_VERSION
        store._header[0]['layout_digest'] = layout.digest.encode('ascii')
        store._header[0]['row_bytes'] = layout.row_bytes
        store._header[0]['maximum_capacity'] = maximum_capacity
        store._header[0]['logical_capacity'] = logical_capacity
        store._header[0]['head'] = 0
        store._header[0]['size'] = 0
        store._header[0]['evicted_rows'] = 0
        store.flush()
        return store

    @property
    def allocated_file_size(self) -> int:
        return self.path.stat().st_size

    @classmethod
    def open(cls, path: Path, layout: ReplayLayout, writable: bool = True) -> ReplayStore:
        if not path.is_file():
            raise ValueError(f'Replay store does not exist: {path}')
        file = path.open('r+b' if writable else 'rb')
        access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        mapping = mmap.mmap(file.fileno(), 0, access=access)
        try:
            header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=mapping)
            maximum_capacity = int(header[0]['maximum_capacity'])
            rows = np.ndarray(
                (maximum_capacity,),
                dtype=layout.row_dtype,
                buffer=mapping,
                offset=_HEADER_DTYPE.itemsize,
            )
            store = cls(path, layout, file, mapping, header, rows, writable)
            store._validate_header()
            return store
        except BaseException:
            mapping.close()
            file.close()
            raise

    @classmethod
    def _map(
        cls,
        path: Path,
        layout: ReplayLayout,
        writable: bool,
        maximum_capacity: int,
    ) -> ReplayStore:
        file = path.open('r+b' if writable else 'rb')
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ)
        header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=mapping)
        rows = np.ndarray(
            (maximum_capacity,),
            dtype=layout.row_dtype,
            buffer=mapping,
            offset=_HEADER_DTYPE.itemsize,
        )
        return cls(path, layout, file, mapping, header, rows, writable)

    @staticmethod
    def _validate_capacities(maximum_capacity: int, logical_capacity: int) -> None:
        if maximum_capacity <= 0:
            raise ValueError('Replay maximum capacity must be positive.')
        if not 1 <= logical_capacity <= maximum_capacity:
            raise ValueError('Replay logical capacity must lie within its maximum capacity.')

    @property
    def state(self) -> ReplayStoreState:
        header = self._header[0]
        return ReplayStoreState(
            maximum_capacity=int(header['maximum_capacity']),
            logical_capacity=int(header['logical_capacity']),
            head=int(header['head']),
            size=int(header['size']),
            evicted_rows=int(header['evicted_rows']),
        )

    def set_logical_capacity(self, logical_capacity: int) -> None:
        self._ensure_writable()
        state = self.state
        self._validate_capacities(state.maximum_capacity, logical_capacity)
        if state.size > logical_capacity:
            removed = state.size - logical_capacity
            self._header[0]['head'] = (state.head + removed) % state.maximum_capacity
            self._header[0]['size'] = logical_capacity
            self._header[0]['evicted_rows'] = state.evicted_rows + removed
        self._header[0]['logical_capacity'] = logical_capacity

    def append(self, sample: ReplaySample) -> None:
        self._ensure_writable()
        encoded_row = np.zeros((1,), dtype=self.layout.row_dtype)
        self._write_sample(encoded_row[0], sample)
        state = self.state
        if state.size == state.logical_capacity:
            physical_index = (state.head + state.size) % state.maximum_capacity
            self._header[0]['head'] = (state.head + 1) % state.maximum_capacity
            self._header[0]['evicted_rows'] = state.evicted_rows + 1
        else:
            physical_index = (state.head + state.size) % state.maximum_capacity
            self._header[0]['size'] = state.size + 1
        self._rows[physical_index] = encoded_row[0]

    def sample_at(self, logical_index: int) -> ReplaySample:
        state = self.state
        if not 0 <= logical_index < state.size:
            raise ValueError(f'Replay row {logical_index} is outside the live FIFO.')
        physical_index = (state.head + logical_index) % state.maximum_capacity
        return self._read_sample(self._rows[physical_index])

    def flush(self) -> None:
        if self._writable:
            self._mapping.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._header = np.empty((0,), dtype=_HEADER_DTYPE)
        self._rows = np.empty((0,), dtype=self.layout.row_dtype)
        self._mapping.close()
        self._file.close()
        self._closed = True

    def _validate_header(self) -> None:
        header = self._header[0]
        if bytes(header['magic']) != _REPLAY_MAGIC:
            raise ValueError('Replay store magic is invalid.')
        if int(header['schema_version']) != _REPLAY_SCHEMA_VERSION:
            raise ValueError('Replay store schema version is unsupported.')
        if bytes(header['layout_digest']).decode('ascii') != self.layout.digest:
            raise ValueError('Replay store layout does not match the experiment.')
        if int(header['row_bytes']) != self.layout.row_bytes:
            raise ValueError('Replay store row width does not match the experiment.')
        state = self.state
        self._validate_capacities(state.maximum_capacity, state.logical_capacity)
        if not 0 <= state.head < state.maximum_capacity or not 0 <= state.size <= state.logical_capacity:
            raise ValueError('Replay store FIFO header is invalid.')
        expected_size = _HEADER_DTYPE.itemsize + state.maximum_capacity * self.layout.row_bytes
        if self.path.stat().st_size != expected_size:
            raise ValueError('Replay store file size does not match its header and layout.')

    def _write_sample(self, row: np.void, sample: ReplaySample) -> None:
        if len(sample.encoded_state) != self.layout.packed_planes.payload_bytes:
            raise ValueError('Replay sample packed state has the wrong width.')
        if len(sample.auxiliary_targets) != len(self.layout.targets.auxiliary_heads):
            raise ValueError('Replay sample auxiliary targets do not match the fixed layout.')
        row['encoded_state'] = np.frombuffer(bytes(sample.encoded_state), dtype=np.uint8)
        self._write_policy(row, 'policy', sample.policy)
        row['wdl_target'] = (sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss)
        row['root_value'] = sample.root_value
        for index, (head, target) in enumerate(zip(self.layout.targets.auxiliary_heads, sample.auxiliary_targets)):
            match head, target:
                case NextPolicyHeadLayout(), EligibleNextPolicyTarget(policy=policy):
                    self._write_policy(row, f'auxiliary_{index}', policy)
                    row[f'auxiliary_{index}_eligible'] = 1
                case NextPolicyHeadLayout(), IneligibleNextPolicyTarget():
                    row[f'auxiliary_{index}_eligible'] = 0
                case RemainingGameLengthHeadLayout(), EligibleRemainingGameLengthTarget(normalized_length=value):
                    row[f'auxiliary_{index}_value'] = value
                    row[f'auxiliary_{index}_eligible'] = 1
                case RemainingGameLengthHeadLayout(), IneligibleRemainingGameLengthTarget():
                    row[f'auxiliary_{index}_eligible'] = 0
                case _:
                    raise ValueError('Replay auxiliary target does not match its fixed layout.')
        row['sample_weight'] = sample.sample_weight
        row['source_model_generation'] = sample.source_model_generation
        row['source_timestamp'] = sample.source_created_at_seconds

    def _write_policy(self, row: np.void, prefix: str, policy: SparsePolicyTarget) -> None:
        if len(policy.visits) > self.layout.maximum_policy_entries:
            raise ValueError('Sparse policy exceeds the configured retained-entry count.')
        if any(visit.action_id >= self.layout.targets.action_size for visit in policy.visits):
            raise ValueError('Sparse policy contains an action outside the action space.')
        if any(visit.visit_count > 65_535 for visit in policy.visits):
            raise ValueError('Sparse policy visit count does not fit uint16.')
        row[f'{prefix}_entry_count'] = len(policy.visits)
        row[f'{prefix}_action_ids'][: len(policy.visits)] = tuple(visit.action_id for visit in policy.visits)
        row[f'{prefix}_visit_counts'][: len(policy.visits)] = tuple(visit.visit_count for visit in policy.visits)

    def _read_sample(self, row: np.void) -> ReplaySample:
        auxiliary_targets = []
        for index, head in enumerate(self.layout.targets.auxiliary_heads):
            match head:
                case NextPolicyHeadLayout():
                    if int(row[f'auxiliary_{index}_eligible']):
                        auxiliary_targets.append(
                            EligibleNextPolicyTarget(policy=self._read_policy(row, f'auxiliary_{index}'))
                        )
                    else:
                        auxiliary_targets.append(IneligibleNextPolicyTarget())
                case RemainingGameLengthHeadLayout():
                    if int(row[f'auxiliary_{index}_eligible']):
                        auxiliary_targets.append(
                            EligibleRemainingGameLengthTarget(normalized_length=float(row[f'auxiliary_{index}_value']))
                        )
                    else:
                        auxiliary_targets.append(IneligibleRemainingGameLengthTarget())
        wdl = row['wdl_target']
        return ReplaySample(
            encoded_state=self.layout.packed_planes.value(bytes(row['encoded_state'])),
            policy=self._read_policy(row, 'policy'),
            wdl_target=WdlTarget(win=float(wdl[0]), draw=float(wdl[1]), loss=float(wdl[2])),
            root_value=float(row['root_value']),
            auxiliary_targets=tuple(auxiliary_targets),
            sample_weight=float(row['sample_weight']),
            source_model_generation=int(row['source_model_generation']),
            source_created_at_seconds=float(row['source_timestamp']),
        )

    @staticmethod
    def _read_policy(row: np.void, prefix: str) -> SparsePolicyTarget:
        count = int(row[f'{prefix}_entry_count'])
        actions = row[f'{prefix}_action_ids'][:count]
        visits = row[f'{prefix}_visit_counts'][:count]
        return SparsePolicyTarget(
            visits=tuple(
                GameSearchVisit(action_id=int(action), visit_count=int(visit_count))
                for action, visit_count in zip(actions, visits)
            )
        )

    def _ensure_writable(self) -> None:
        if self._closed:
            raise RuntimeError('Replay store is closed.')
        if not self._writable:
            raise RuntimeError('Replay store is read-only.')
