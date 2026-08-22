from __future__ import annotations

import hashlib
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
from src.games.contracts import WdlTarget
from src.replay.columnar import (
    ReplayColumnArray,
    ReplayColumnViews,
    ReplayLegalMovesColumnViews,
    ReplayNextPolicyColumnViews,
    ReplayPolicyColumnViews,
    ReplayScalarColumnViews,
    ReplaySearchCorrectionColumnViews,
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
from src.replay.layout import ReplayColumnDescriptor, ReplayLayout
from src.self_play.completed_game import SearchVisitCounts
from src.training.targets import (
    FutureSearchValueHeadLayout,
    IrreversibleProgressHeadLayout,
    LegalMovesHeadLayout,
    NextPolicyHeadLayout,
    RemainingGameLengthHeadLayout,
    SearchCorrectionHeadLayout,
)

_REPLAY_MAGIC = b'AZRPLY02'
_REPLAY_SCHEMA_VERSION = 4
_REPLAY_ENDIAN_MARKER = 0x0102
_REPLAY_CONTAINER_STORE = 1
_HEADER_BYTES = 65_536
_COLUMN_ALIGNMENT = 4_096
_DESCRIPTOR_TABLE_OFFSET = 512
_MAXIMUM_COLUMN_COUNT = 128
_TRANSACTION_IDENTITY_BYTES = 64

_HEADER_DTYPE = np.dtype(
    [
        ('magic', 'S8'),
        ('schema_version', '<u2'),
        ('endian_marker', '<u2'),
        ('container_kind', 'u1'),
        ('reserved', 'u1', (3,)),
        ('header_bytes', '<u4'),
        ('descriptor_count', '<u2'),
        ('descriptor_bytes', '<u2'),
        ('layout_digest', 'S64'),
        ('descriptor_digest', 'S64'),
        ('maximum_capacity', '<u8'),
        ('logical_capacity', '<u8'),
        ('head', '<u8'),
        ('size', '<u8'),
        ('evicted_rows', '<u8'),
        ('total_appended_rows', '<u8'),
        ('append_sequence', '<u8'),
        ('last_transaction_identity', f'S{_TRANSACTION_IDENTITY_BYTES}'),
        ('last_transaction_row_count', '<u8'),
    ],
    align=False,
)
_COLUMN_DESCRIPTOR_DTYPE = np.dtype(
    [
        ('name', 'S64'),
        ('dtype', 'S8'),
        ('rank', 'u1'),
        ('reserved', 'u1', (7,)),
        ('dimensions', '<u4', (4,)),
        ('offset', '<u8'),
        ('row_bytes', '<u4'),
        ('slab_bytes', '<u8'),
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
    total_appended_rows: int
    append_sequence: int
    last_transaction_identity: str
    last_transaction_row_count: int


@dataclass(frozen=True)
class ReplayAppendPlan:
    row_count: int
    transaction_identity: str
    before: ReplayStoreState
    after: ReplayStoreState


@dataclass(frozen=True)
class ReplayAppendTransaction:
    row_count: int
    transaction_identity: str


@dataclass(frozen=True)
class ReplayPhysicalColumn:
    descriptor: ReplayColumnDescriptor
    offset: int
    slab_bytes: int


class ReplayStore:
    def __init__(
        self,
        path: Path,
        layout: ReplayLayout,
        file: BinaryIO,
        mapping: mmap.mmap,
        header: npt.NDArray[np.void],
        descriptor_table: npt.NDArray[np.void],
        column_arrays: tuple[ReplayColumnArray, ...],
        writable: bool,
    ) -> None:
        self.path = path
        self.layout = layout
        self._file = file
        self._mapping = mapping
        self._header = header
        self._descriptor_table = descriptor_table
        self._column_arrays = column_arrays
        self._writable = writable
        self._closed = False

    @classmethod
    def projected_file_size(cls, layout: ReplayLayout, maximum_capacity: int) -> int:
        cls._validate_capacities(maximum_capacity, maximum_capacity)
        columns = _physical_columns(layout, maximum_capacity)
        return _HEADER_BYTES if not columns else columns[-1].offset + columns[-1].slab_bytes

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
        store._initialize_header(maximum_capacity, logical_capacity)
        store.flush()
        return store

    @property
    def allocated_file_size(self) -> int:
        return self.path.stat().st_size

    @property
    def physical_columns(self) -> tuple[ReplayPhysicalColumn, ...]:
        return _physical_columns(self.layout, self.state.maximum_capacity)

    @classmethod
    def open(cls, path: Path, layout: ReplayLayout, writable: bool = True) -> ReplayStore:
        return cls._open(path, layout, writable, validate_fifo=True)

    @classmethod
    def open_for_recovery(cls, path: Path, layout: ReplayLayout) -> ReplayStore:
        return cls._open(path, layout, writable=True, validate_fifo=False)

    @classmethod
    def _open(
        cls,
        path: Path,
        layout: ReplayLayout,
        writable: bool,
        validate_fifo: bool,
    ) -> ReplayStore:
        if not path.is_file():
            raise ValueError(f'Replay store does not exist: {path}')
        if path.stat().st_size < _HEADER_BYTES:
            raise ValueError('Replay store is smaller than the fixed schema-4 header.')
        file = path.open('r+b' if writable else 'rb')
        access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        mapping: mmap.mmap | None = None
        try:
            mapping = mmap.mmap(file.fileno(), 0, access=access)
            header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=mapping)
            if bytes(header[0]['magic']) != _REPLAY_MAGIC:
                raise ValueError('Replay store magic is invalid or belongs to an unsupported schema.')
            if int(header[0]['schema_version']) != _REPLAY_SCHEMA_VERSION:
                raise ValueError('Replay store schema version is unsupported.')
            if int(header[0]['header_bytes']) != _HEADER_BYTES:
                raise ValueError('Replay store header width is invalid.')
            maximum_capacity = int(header[0]['maximum_capacity'])
            cls._validate_capacities(maximum_capacity, maximum_capacity)
            if path.stat().st_size != cls.projected_file_size(layout, maximum_capacity):
                raise ValueError('Replay store file size does not match its header and layout.')
            del header
            mapping.close()
            file.close()
            store = cls._map(path, layout, writable=writable, maximum_capacity=maximum_capacity)
            try:
                store._validate_header(validate_fifo)
            except BaseException:
                store._release_mapping()
                raise
            return store
        except BaseException:
            if mapping is not None and not mapping.closed:
                mapping.close()
            if not file.closed:
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
        columns = _physical_columns(layout, maximum_capacity)
        file = path.open('r+b' if writable else 'rb')
        mapping: mmap.mmap | None = None
        header: npt.NDArray[np.void] | None = None
        descriptor_table: npt.NDArray[np.void] | None = None
        arrays: tuple[ReplayColumnArray, ...] = ()
        try:
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ)
            header = np.ndarray((1,), dtype=_HEADER_DTYPE, buffer=mapping)
            descriptor_table = np.ndarray(
                (_MAXIMUM_COLUMN_COUNT,),
                dtype=_COLUMN_DESCRIPTOR_DTYPE,
                buffer=mapping,
                offset=_DESCRIPTOR_TABLE_OFFSET,
            )
            arrays = tuple(
                ReplayColumnArray(
                    column.descriptor,
                    np.ndarray(
                        (maximum_capacity, *column.descriptor.trailing_shape),
                        dtype=column.descriptor.element_type.numpy_dtype,
                        buffer=mapping,
                        offset=column.offset,
                    ),
                )
                for column in columns
            )
            return cls(path, layout, file, mapping, header, descriptor_table, arrays, writable)
        except BaseException:
            arrays = ()
            descriptor_table = None
            header = None
            if mapping is not None:
                mapping.close()
            file.close()
            raise

    def _initialize_header(self, maximum_capacity: int, logical_capacity: int) -> None:
        self._mapping[:_HEADER_BYTES] = bytes(_HEADER_BYTES)
        header = self._header[0]
        header['magic'] = _REPLAY_MAGIC
        header['schema_version'] = _REPLAY_SCHEMA_VERSION
        header['endian_marker'] = _REPLAY_ENDIAN_MARKER
        header['container_kind'] = _REPLAY_CONTAINER_STORE
        header['header_bytes'] = _HEADER_BYTES
        header['descriptor_count'] = len(self._column_arrays)
        header['descriptor_bytes'] = _COLUMN_DESCRIPTOR_DTYPE.itemsize
        header['layout_digest'] = self.layout.digest.encode('ascii')
        header['maximum_capacity'] = maximum_capacity
        header['logical_capacity'] = logical_capacity
        for index, physical in enumerate(_physical_columns(self.layout, maximum_capacity)):
            row = self._descriptor_table[index]
            row['name'] = physical.descriptor.key.name.encode('ascii')
            row['dtype'] = physical.descriptor.element_type.value.encode('ascii')
            row['rank'] = len(physical.descriptor.trailing_shape)
            row['dimensions'][: len(physical.descriptor.trailing_shape)] = physical.descriptor.trailing_shape
            row['offset'] = physical.offset
            row['row_bytes'] = physical.descriptor.row_bytes
            row['slab_bytes'] = physical.slab_bytes
        header['descriptor_digest'] = self._descriptor_digest().encode('ascii')

    @staticmethod
    def _validate_capacities(maximum_capacity: int, logical_capacity: int) -> None:
        if maximum_capacity <= 0:
            raise ValueError('Replay maximum capacity must be positive.')
        if not 1 <= logical_capacity <= maximum_capacity:
            raise ValueError('Replay logical capacity must lie within its maximum capacity.')

    @property
    def total_appended_rows(self) -> int:
        return self.state.total_appended_rows

    @property
    def state(self) -> ReplayStoreState:
        header = self._header[0]
        transaction_bytes = bytes(header['last_transaction_identity'])
        return ReplayStoreState(
            maximum_capacity=int(header['maximum_capacity']),
            logical_capacity=int(header['logical_capacity']),
            head=int(header['head']),
            size=int(header['size']),
            evicted_rows=int(header['evicted_rows']),
            total_appended_rows=int(header['total_appended_rows']),
            append_sequence=int(header['append_sequence']),
            last_transaction_identity=_decode_ascii(transaction_bytes, 'transaction identity'),
            last_transaction_row_count=int(header['last_transaction_row_count']),
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
        self.extend((sample,))

    def extend(self, samples: tuple[ReplaySample, ...]) -> None:
        self.extend_rows(encode_replay_rows(self.layout, samples))

    def extend_rows(
        self,
        rows: npt.NDArray[np.void],
        transaction_identity: str = '',
    ) -> None:
        if rows.dtype != self.layout.row_dtype:
            raise ValueError('Replay rows do not match the store row layout.')
        self.append_columns(_column_views_from_rows(self.layout, rows), transaction_identity)

    def append_columns(
        self,
        columns: ReplayColumnViews,
        transaction_identity: str = '',
    ) -> None:
        self._ensure_writable()
        _transaction_bytes(transaction_identity)
        if transaction_identity and self.state.last_transaction_identity == transaction_identity:
            if self.state.last_transaction_row_count != columns.row_count:
                raise ValueError('Committed replay transaction identity has a different row count.')
            return
        plan = self.plan_append(columns.row_count, transaction_identity)
        self.apply_append_plan(columns, plan)

    def plan_append(self, row_count: int, transaction_identity: str) -> ReplayAppendPlan:
        self._ensure_writable()
        return plan_replay_append_chain(
            self.state,
            (ReplayAppendTransaction(row_count=row_count, transaction_identity=transaction_identity),),
        )[0]

    def apply_append_plan(
        self,
        columns: ReplayColumnViews,
        plan: ReplayAppendPlan,
    ) -> None:
        self.apply_append_plan_slices((columns,), plan)

    def apply_append_plan_slices(
        self,
        column_slices: tuple[ReplayColumnViews, ...],
        plan: ReplayAppendPlan,
    ) -> None:
        self._ensure_writable()
        flattened_slices = self._validate_append_plan_slices(column_slices, plan)
        current = self.state
        if current == plan.after:
            return
        if current != plan.before:
            raise ValueError('Replay append plan cannot be applied to the current store state.')
        self._apply_column_slices_for_plan(flattened_slices, plan)

    def reapply_append_plan(
        self,
        columns: ReplayColumnViews,
        plan: ReplayAppendPlan,
    ) -> None:
        self.reapply_append_plan_slices((columns,), plan)

    def reapply_append_plan_slices(
        self,
        column_slices: tuple[ReplayColumnViews, ...],
        plan: ReplayAppendPlan,
    ) -> None:
        self._ensure_writable()
        flattened_slices = self._validate_append_plan_slices(column_slices, plan)
        current = self.state
        if current == plan.after:
            return
        if not _is_interrupted_append_state(current, plan):
            raise ValueError('Replay append recovery found an ambiguous store state.')
        self._apply_column_slices_for_plan(flattened_slices, plan)

    def _validate_append_plan_slices(
        self,
        column_slices: tuple[ReplayColumnViews, ...],
        plan: ReplayAppendPlan,
    ) -> tuple[tuple[ReplayColumnArray, ...], ...]:
        _validate_planning_state(plan.before)
        _transaction_bytes(plan.transaction_identity)
        if plan.row_count != sum(columns.row_count for columns in column_slices):
            raise ValueError('Replay append plan row count does not match its column slices.')
        if plan.after != _append_state(plan.before, plan.row_count, plan.transaction_identity):
            raise ValueError('Replay append plan has an invalid final state.')
        flattened_slices = []
        for columns in column_slices:
            source_arrays = flatten_column_views(self.layout, columns)
            self._validate_column_arrays(source_arrays, columns.row_count)
            self._validate_column_semantics(columns)
            flattened_slices.append(source_arrays)
        return tuple(flattened_slices)

    def _apply_column_slices_for_plan(
        self,
        flattened_slices: tuple[tuple[ReplayColumnArray, ...], ...],
        plan: ReplayAppendPlan,
    ) -> None:
        row_count = plan.row_count
        state = plan.before
        write_count = min(row_count, state.logical_capacity)
        retained_start = row_count - write_count
        old_tail = (state.head + state.size) % state.maximum_capacity
        slice_start = 0
        for source_arrays in flattened_slices:
            slice_row_count = len(source_arrays[0].values) if source_arrays else 0
            slice_end = slice_start + slice_row_count
            retained_slice_start = max(slice_start, retained_start)
            if retained_slice_start < slice_end:
                source_start = retained_slice_start - slice_start
                copy_count = slice_end - retained_slice_start
                destination_start = (old_tail + retained_slice_start) % state.maximum_capacity
                first_count = min(copy_count, state.maximum_capacity - destination_start)
                for destination, source in zip(self._column_arrays, source_arrays, strict=True):
                    destination.values[destination_start : destination_start + first_count] = source.values[
                        source_start : source_start + first_count
                    ]
                    second_count = copy_count - first_count
                    if second_count:
                        destination.values[:second_count] = source.values[
                            source_start + first_count : source_start + copy_count
                        ]
            slice_start = slice_end
        self._write_state(plan.after)

    def logical_to_physical(
        self,
        logical_indices: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        indices = np.asarray(logical_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError('Replay logical indices must be one-dimensional.')
        state = self.state
        if np.any(indices < 0) or np.any(indices >= state.size):
            raise ValueError('Replay logical index is outside the live FIFO.')
        return (state.head + indices) % state.maximum_capacity

    def gather_physical(self, physical_indices: npt.ArrayLike) -> ReplayColumnViews:
        indices = np.asarray(physical_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError('Replay physical indices must be one-dimensional.')
        maximum_capacity = self.state.maximum_capacity
        if np.any(indices < 0) or np.any(indices >= maximum_capacity):
            raise ValueError('Replay physical index is outside the allocated store.')
        gathered = tuple(
            ReplayColumnArray(column.descriptor, np.asarray(column.values[indices])) for column in self._column_arrays
        )
        return build_column_views(self.layout, gathered)

    def gather_logical(self, logical_indices: npt.ArrayLike) -> ReplayColumnViews:
        return self.gather_physical(self.logical_to_physical(logical_indices))

    def sample_at(self, logical_index: int) -> ReplaySample:
        return self._read_sample(self.gather_logical(np.asarray([logical_index], dtype=np.int64)))

    def flush(self) -> None:
        if self._writable:
            self._mapping.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._release_mapping()

    def _release_mapping(self) -> None:
        self._column_arrays = ()
        self._descriptor_table = np.empty((0,), dtype=_COLUMN_DESCRIPTOR_DTYPE)
        self._header = np.empty((0,), dtype=_HEADER_DTYPE)
        self._mapping.close()
        self._file.close()
        self._closed = True

    def _validate_header(self, validate_fifo: bool) -> None:
        header = self._header[0]
        if bytes(header['magic']) != _REPLAY_MAGIC:
            raise ValueError('Replay store magic is invalid.')
        if int(header['schema_version']) != _REPLAY_SCHEMA_VERSION:
            raise ValueError('Replay store schema version is unsupported.')
        if int(header['endian_marker']) != _REPLAY_ENDIAN_MARKER:
            raise ValueError('Replay store endian marker is invalid.')
        if int(header['container_kind']) != _REPLAY_CONTAINER_STORE:
            raise ValueError('Replay container is not a live store.')
        if int(header['header_bytes']) != _HEADER_BYTES:
            raise ValueError('Replay store header width is invalid.')
        if int(header['descriptor_count']) != len(self.layout.columns.columns):
            raise ValueError('Replay store column count does not match the experiment.')
        if int(header['descriptor_bytes']) != _COLUMN_DESCRIPTOR_DTYPE.itemsize:
            raise ValueError('Replay store column descriptor width is unsupported.')
        if _decode_ascii(bytes(header['layout_digest']), 'layout digest') != self.layout.digest:
            raise ValueError('Replay store layout does not match the experiment.')
        if _decode_ascii(bytes(header['descriptor_digest']), 'descriptor digest') != self._descriptor_digest():
            raise ValueError('Replay store column descriptors are invalid.')
        state = self.state
        self._validate_capacities(state.maximum_capacity, state.logical_capacity)
        self._validate_descriptor_table(state.maximum_capacity)
        if validate_fifo:
            if not 0 <= state.head < state.maximum_capacity or not 0 <= state.size <= state.logical_capacity:
                raise ValueError('Replay store FIFO header is invalid.')
            if state.evicted_rows + state.size != state.total_appended_rows:
                raise ValueError('Replay store append counters are invalid.')
            if state.last_transaction_row_count > state.total_appended_rows or (
                state.append_sequence == 0
                and (
                    state.total_appended_rows != 0
                    or state.last_transaction_identity
                    or state.last_transaction_row_count != 0
                )
            ):
                raise ValueError('Replay store transaction counters are invalid.')
        expected_size = self.projected_file_size(self.layout, state.maximum_capacity)
        if self.path.stat().st_size != expected_size:
            raise ValueError('Replay store file size does not match its header and layout.')

    def _descriptor_digest(self) -> str:
        count = len(self.layout.columns.columns)
        return hashlib.sha256(self._descriptor_table[:count].tobytes()).hexdigest()

    def _validate_descriptor_table(self, maximum_capacity: int) -> None:
        for row, physical in zip(
            self._descriptor_table,
            _physical_columns(self.layout, maximum_capacity),
            strict=False,
        ):
            descriptor = physical.descriptor
            dimensions = tuple(int(value) for value in row['dimensions'][: int(row['rank'])])
            if (
                _decode_ascii(bytes(row['name']), 'column name') != descriptor.key.name
                or _decode_ascii(bytes(row['dtype']), 'column dtype') != descriptor.element_type.value
                or int(row['rank']) != len(descriptor.trailing_shape)
                or dimensions != descriptor.trailing_shape
                or int(row['offset']) != physical.offset
                or int(row['row_bytes']) != descriptor.row_bytes
                or int(row['slab_bytes']) != physical.slab_bytes
            ):
                raise ValueError(f'Replay column descriptor is invalid: {descriptor.key.name}.')

    def _validate_column_arrays(
        self,
        arrays: tuple[ReplayColumnArray, ...],
        row_count: int,
    ) -> None:
        if len(arrays) != len(self._column_arrays):
            raise ValueError('Replay columns do not match the fixed layout.')
        for expected, actual in zip(self._column_arrays, arrays, strict=True):
            if expected.descriptor != actual.descriptor:
                raise ValueError('Replay column descriptor does not match the fixed layout.')
            expected_shape = (row_count, *expected.descriptor.trailing_shape)
            if actual.values.shape != expected_shape:
                raise ValueError(f'Replay column {expected.descriptor.key.name} has the wrong shape.')
            if actual.values.dtype != expected.descriptor.element_type.numpy_dtype:
                raise ValueError(f'Replay column {expected.descriptor.key.name} has the wrong dtype.')
        for target in columns_with_eligibility(build_column_views(self.layout, arrays)):
            if np.any((target != 0) & (target != 1)):
                raise ValueError('Replay eligibility columns must contain only zero or one.')

    def _validate_column_semantics(self, columns: ReplayColumnViews) -> None:
        active_rows = np.ones(columns.row_count, dtype=np.bool_)
        _validate_policy_columns(
            columns.policy,
            active_rows,
            self.layout.targets.action_size,
            self.layout.maximum_policy_entries,
            self.layout.maximum_legal_actions,
        )
        wdl = columns.wdl_target
        if np.any(~np.isfinite(wdl)) or np.any(wdl < 0.0) or np.any(np.abs(wdl.sum(axis=1) - 1.0) > 1e-6):
            raise ValueError('Replay WDL targets must be finite, nonnegative, and sum to one.')
        if np.any(~np.isfinite(columns.root_value)) or np.any(np.abs(columns.root_value) > 1.0):
            raise ValueError('Replay root values must be finite and lie in [-1, 1].')
        if np.any(~np.isfinite(columns.sample_weight)) or np.any(columns.sample_weight <= 0.0):
            raise ValueError('Replay sample weights must be finite and positive.')
        if np.any(~np.isfinite(columns.source_timestamp)) or np.any(columns.source_timestamp < 0.0):
            raise ValueError('Replay source timestamps must be finite and nonnegative.')
        for target in columns.auxiliary:
            match target:
                case ReplayNextPolicyColumnViews():
                    _validate_policy_columns(
                        target.policy,
                        target.eligible.astype(np.bool_),
                        self.layout.targets.action_size,
                        self.layout.maximum_policy_entries,
                        self.layout.maximum_legal_actions,
                    )
                case ReplayScalarColumnViews(kind='remaining_game_length'):
                    _validate_eligible_values(target.value, target.eligible, minimum=0.0, maximum=None)
                case ReplayScalarColumnViews(kind='future_search_value'):
                    _validate_eligible_values(target.value, target.eligible, minimum=-1.0, maximum=1.0)
                case ReplayScalarColumnViews(kind='irreversible_progress'):
                    _validate_eligible_values(target.value, target.eligible, minimum=0.0, maximum=1.0)
                case ReplaySearchCorrectionColumnViews():
                    if np.any(~np.isfinite(target.value)) or np.any(target.value < 0.0) or np.any(target.value > 1.0):
                        raise ValueError('Replay search-correction targets must be finite and lie in [0, 1].')
                case ReplayLegalMovesColumnViews():
                    pass

    def _write_state(self, state: ReplayStoreState) -> None:
        header = self._header[0]
        header['maximum_capacity'] = state.maximum_capacity
        header['logical_capacity'] = state.logical_capacity
        header['head'] = state.head
        header['size'] = state.size
        header['evicted_rows'] = state.evicted_rows
        header['total_appended_rows'] = state.total_appended_rows
        header['append_sequence'] = state.append_sequence
        header['last_transaction_identity'] = _transaction_bytes(state.last_transaction_identity)
        header['last_transaction_row_count'] = state.last_transaction_row_count

    def _read_sample(self, columns: ReplayColumnViews) -> ReplaySample:
        auxiliary_targets = []
        for head, target in zip(self.layout.targets.auxiliary_heads, columns.auxiliary, strict=True):
            match head, target:
                case NextPolicyHeadLayout(), ReplayNextPolicyColumnViews():
                    if int(target.eligible[0]):
                        auxiliary_targets.append(EligibleNextPolicyTarget(policy=self._read_policy(target.policy)))
                    else:
                        auxiliary_targets.append(IneligibleNextPolicyTarget())
                case RemainingGameLengthHeadLayout(), ReplayScalarColumnViews():
                    if int(target.eligible[0]):
                        auxiliary_targets.append(
                            EligibleRemainingGameLengthTarget(normalized_length=float(target.value[0]))
                        )
                    else:
                        auxiliary_targets.append(IneligibleRemainingGameLengthTarget())
                case (
                    (FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout()),
                    ReplayScalarColumnViews(),
                ):
                    if int(target.eligible[0]):
                        auxiliary_targets.append(
                            EligibleScalarAuxiliaryTarget(kind=head.kind, value=float(target.value[0]))
                        )
                    else:
                        auxiliary_targets.append(IneligibleScalarAuxiliaryTarget(kind=head.kind))
                case SearchCorrectionHeadLayout(), ReplaySearchCorrectionColumnViews():
                    auxiliary_targets.append(
                        EligibleScalarAuxiliaryTarget(kind='search_correction', value=float(target.value[0]))
                    )
                case LegalMovesHeadLayout(), ReplayLegalMovesColumnViews():
                    auxiliary_targets.append(EligibleLegalMovesTarget())
                case _:
                    raise ValueError('Replay auxiliary columns do not match the fixed layout.')
        wdl = columns.wdl_target[0]
        return ReplaySample(
            encoded_state=self.layout.packed_planes.value(columns.encoded_state[0].tobytes()),
            policy=self._read_policy(columns.policy),
            wdl_target=WdlTarget(win=float(wdl[0]), draw=float(wdl[1]), loss=float(wdl[2])),
            root_value=float(columns.root_value[0]),
            auxiliary_targets=tuple(auxiliary_targets),
            sample_weight=float(columns.sample_weight[0]),
            source_model_generation=int(columns.source_model_generation[0]),
            source_created_at_seconds=float(columns.source_timestamp[0]),
        )

    @staticmethod
    def _read_policy(policy: ReplayPolicyColumnViews) -> SparsePolicyTarget:
        count = int(policy.entry_count[0])
        legal_count = int(policy.legal_count[0])
        return SparsePolicyTarget(
            visits=SearchVisitCounts(
                action_ids=tuple(int(action) for action in policy.action_ids[0, :count]),
                visit_counts=tuple(int(visit_count) for visit_count in policy.visit_counts[0, :count]),
            ),
            legal_action_ids=tuple(int(action_id) for action_id in policy.legal_action_ids[0, :legal_count]),
        )

    def _ensure_writable(self) -> None:
        if self._closed:
            raise RuntimeError('Replay store is closed.')
        if not self._writable:
            raise RuntimeError('Replay store is read-only.')


def columns_with_eligibility(columns: ReplayColumnViews) -> tuple[npt.NDArray[np.uint8], ...]:
    return tuple(
        target.eligible
        for target in columns.auxiliary
        if isinstance(target, ReplayNextPolicyColumnViews | ReplayScalarColumnViews)
    )


def encode_replay_rows(layout: ReplayLayout, samples: tuple[ReplaySample, ...]) -> npt.NDArray[np.void]:
    encoded_rows = np.zeros((len(samples),), dtype=layout.row_dtype)
    for row, sample in zip(encoded_rows, samples, strict=True):
        _encode_sample(layout, row, sample)
    return encoded_rows


def _encode_sample(layout: ReplayLayout, row: np.void, sample: ReplaySample) -> None:
    if len(sample.encoded_state) != layout.packed_planes.payload_bytes:
        raise ValueError('Replay sample packed state has the wrong width.')
    if len(sample.auxiliary_targets) != len(layout.targets.auxiliary_heads):
        raise ValueError('Replay sample auxiliary targets do not match the fixed layout.')
    row['encoded_state'] = np.frombuffer(bytes(sample.encoded_state), dtype=np.uint8)
    _encode_policy(layout, row, 'policy', sample.policy)
    row['wdl_target'] = (sample.wdl_target.win, sample.wdl_target.draw, sample.wdl_target.loss)
    row['root_value'] = sample.root_value
    for index, (head, target) in enumerate(zip(layout.targets.auxiliary_heads, sample.auxiliary_targets, strict=True)):
        match head, target:
            case NextPolicyHeadLayout(), EligibleNextPolicyTarget(policy=policy):
                _encode_policy(layout, row, f'auxiliary_{index}', policy)
                row[f'auxiliary_{index}_eligible'] = 1
            case NextPolicyHeadLayout(), IneligibleNextPolicyTarget():
                row[f'auxiliary_{index}_eligible'] = 0
            case RemainingGameLengthHeadLayout(), EligibleRemainingGameLengthTarget(normalized_length=value):
                row[f'auxiliary_{index}_value'] = value
                row[f'auxiliary_{index}_eligible'] = 1
            case RemainingGameLengthHeadLayout(), IneligibleRemainingGameLengthTarget():
                row[f'auxiliary_{index}_eligible'] = 0
            case (
                (FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout()),
                EligibleScalarAuxiliaryTarget(value=value),
            ):
                row[f'auxiliary_{index}_value'] = value
                row[f'auxiliary_{index}_eligible'] = 1
            case (
                (FutureSearchValueHeadLayout() | IrreversibleProgressHeadLayout()),
                IneligibleScalarAuxiliaryTarget(),
            ):
                row[f'auxiliary_{index}_eligible'] = 0
            case SearchCorrectionHeadLayout(), EligibleScalarAuxiliaryTarget(value=value):
                row[f'auxiliary_{index}_value'] = value
            case LegalMovesHeadLayout(), EligibleLegalMovesTarget():
                pass
            case _:
                raise ValueError('Replay auxiliary target does not match its fixed layout.')
    row['sample_weight'] = sample.sample_weight
    row['source_model_generation'] = sample.source_model_generation
    row['source_timestamp'] = sample.source_created_at_seconds


def _encode_policy(layout: ReplayLayout, row: np.void, prefix: str, policy: SparsePolicyTarget) -> None:
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
    row[f'{prefix}_entry_count'] = entry_count
    row[f'{prefix}_action_ids'][:entry_count] = policy.visits.action_ids
    row[f'{prefix}_visit_counts'][:entry_count] = policy.visits.visit_counts
    row[f'{prefix}_legal_count'] = len(policy.legal_action_ids)
    row[f'{prefix}_legal_action_ids'][: len(policy.legal_action_ids)] = policy.legal_action_ids


def _column_views_from_rows(layout: ReplayLayout, rows: npt.NDArray[np.void]) -> ReplayColumnViews:
    arrays = tuple(
        ReplayColumnArray(descriptor, np.asarray(rows[descriptor.key.name])) for descriptor in layout.columns.columns
    )
    return build_column_views(layout, arrays)


def plan_replay_append_chain(
    starting_state: ReplayStoreState,
    transactions: tuple[ReplayAppendTransaction, ...],
) -> tuple[ReplayAppendPlan, ...]:
    _validate_planning_state(starting_state)
    identities = {starting_state.last_transaction_identity} if starting_state.last_transaction_identity else set()
    plans = []
    before = starting_state
    for transaction in transactions:
        if transaction.row_count < 0:
            raise ValueError('Replay append row count must be nonnegative.')
        _transaction_bytes(transaction.transaction_identity)
        if transaction.transaction_identity and transaction.transaction_identity in identities:
            raise ValueError('Replay append transaction identity is already present in the chain.')
        after = _append_state(before, transaction.row_count, transaction.transaction_identity)
        plans.append(
            ReplayAppendPlan(
                row_count=transaction.row_count,
                transaction_identity=transaction.transaction_identity,
                before=before,
                after=after,
            )
        )
        if transaction.transaction_identity:
            identities.add(transaction.transaction_identity)
        before = after
    return tuple(plans)


def _validate_planning_state(state: ReplayStoreState) -> None:
    if state.maximum_capacity <= 0:
        raise ValueError('Replay maximum capacity must be positive.')
    if not 1 <= state.logical_capacity <= state.maximum_capacity:
        raise ValueError('Replay logical capacity must lie within its maximum capacity.')
    if not 0 <= state.head < state.maximum_capacity or not 0 <= state.size <= state.logical_capacity:
        raise ValueError('Replay starting FIFO state is invalid.')
    if state.evicted_rows < 0 or state.total_appended_rows < 0 or state.append_sequence < 0:
        raise ValueError('Replay starting counters must be nonnegative.')
    if state.evicted_rows + state.size != state.total_appended_rows:
        raise ValueError('Replay starting append counters are invalid.')
    _transaction_bytes(state.last_transaction_identity)
    if state.last_transaction_row_count < 0 or state.last_transaction_row_count > state.total_appended_rows:
        raise ValueError('Replay starting transaction counters are invalid.')
    if state.append_sequence == 0 and (
        state.total_appended_rows != 0 or state.last_transaction_identity or state.last_transaction_row_count != 0
    ):
        raise ValueError('Replay starting transaction counters are invalid.')


def _append_state(
    before: ReplayStoreState,
    row_count: int,
    transaction_identity: str,
) -> ReplayStoreState:
    evicted_rows = max(0, before.size + row_count - before.logical_capacity)
    return ReplayStoreState(
        maximum_capacity=before.maximum_capacity,
        logical_capacity=before.logical_capacity,
        head=(before.head + evicted_rows) % before.maximum_capacity,
        size=min(before.logical_capacity, before.size + row_count),
        evicted_rows=before.evicted_rows + evicted_rows,
        total_appended_rows=before.total_appended_rows + row_count,
        append_sequence=before.append_sequence + 1,
        last_transaction_identity=transaction_identity,
        last_transaction_row_count=row_count,
    )


def _is_interrupted_append_state(current: ReplayStoreState, plan: ReplayAppendPlan) -> bool:
    before = plan.before
    after = plan.after
    return (
        current.maximum_capacity == before.maximum_capacity
        and current.logical_capacity == before.logical_capacity
        and current.head in {before.head, after.head}
        and current.size in {before.size, after.size}
        and current.evicted_rows in {before.evicted_rows, after.evicted_rows}
        and current.total_appended_rows in {before.total_appended_rows, after.total_appended_rows}
        and current.append_sequence in {before.append_sequence, after.append_sequence}
        and current.last_transaction_identity in {before.last_transaction_identity, after.last_transaction_identity}
        and current.last_transaction_row_count in {before.last_transaction_row_count, after.last_transaction_row_count}
    )


def _transaction_bytes(transaction_identity: str) -> bytes:
    try:
        encoded_identity = transaction_identity.encode('ascii')
    except UnicodeEncodeError as error:
        raise ValueError('Replay transaction identity must contain only ASCII characters.') from error
    if len(encoded_identity) > _TRANSACTION_IDENTITY_BYTES:
        raise ValueError('Replay transaction identity is too long.')
    return encoded_identity


def _decode_ascii(encoded: bytes, field: str) -> str:
    try:
        return encoded.rstrip(b'\x00').decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError(f'Replay store {field} is not valid ASCII.') from error


def _validate_policy_columns(
    policy: ReplayPolicyColumnViews,
    active_rows: npt.NDArray[np.bool_],
    action_size: int,
    maximum_policy_entries: int,
    maximum_legal_actions: int,
) -> None:
    if not np.any(active_rows):
        return
    if np.any(policy.entry_count[active_rows] == 0) or np.any(policy.entry_count[active_rows] > maximum_policy_entries):
        raise ValueError('Active replay policies require a valid nonempty entry count.')
    if np.any(policy.legal_count[active_rows] == 0) or np.any(policy.legal_count[active_rows] > maximum_legal_actions):
        raise ValueError('Active replay policies require a valid nonempty legal-action count.')
    entry_positions = np.arange(maximum_policy_entries, dtype=np.int64)[np.newaxis, :]
    legal_positions = np.arange(maximum_legal_actions, dtype=np.int64)[np.newaxis, :]
    active_entries = active_rows[:, np.newaxis] & (entry_positions < policy.entry_count[:, np.newaxis])
    active_legal = active_rows[:, np.newaxis] & (legal_positions < policy.legal_count[:, np.newaxis])
    action_ids = policy.action_ids.astype(np.int64, copy=False)
    legal_action_ids = policy.legal_action_ids.astype(np.int64, copy=False)
    if np.any(action_ids[active_entries] >= action_size) or np.any(legal_action_ids[active_legal] >= action_size):
        raise ValueError('Replay policy action IDs must lie inside the action space.')
    if np.any(policy.visit_counts[active_entries] == 0):
        raise ValueError('Replay policy visit counts must be positive.')
    row_numbers = np.arange(policy.entry_count.shape[0], dtype=np.int64)[:, np.newaxis]
    visited_keys = (row_numbers * action_size + action_ids)[active_entries]
    legal_keys = (row_numbers * action_size + legal_action_ids)[active_legal]
    if np.unique(visited_keys).size != visited_keys.size:
        raise ValueError('Replay policy visited action IDs must be unique.')
    if np.unique(legal_keys).size != legal_keys.size:
        raise ValueError('Replay policy legal action IDs must be unique.')
    if np.any(~np.isin(visited_keys, legal_keys)):
        raise ValueError('Replay policy visited actions must be legal.')


def _validate_eligible_values(
    values: npt.NDArray[np.float32],
    eligibility: npt.NDArray[np.uint8],
    minimum: float,
    maximum: float | None,
) -> None:
    eligible_values = values[eligibility.astype(np.bool_)]
    if np.any(~np.isfinite(eligible_values)) or np.any(eligible_values < minimum):
        raise ValueError('Eligible replay scalar targets are outside their valid range.')
    if maximum is not None and np.any(eligible_values > maximum):
        raise ValueError('Eligible replay scalar targets are outside their valid range.')


def _physical_columns(layout: ReplayLayout, maximum_capacity: int) -> tuple[ReplayPhysicalColumn, ...]:
    if len(layout.columns.columns) > _MAXIMUM_COLUMN_COUNT:
        raise ValueError('Replay layout has too many columns for the fixed header.')
    columns: list[ReplayPhysicalColumn] = []
    offset = _HEADER_BYTES
    for descriptor in layout.columns.columns:
        offset = _aligned(offset, _COLUMN_ALIGNMENT)
        slab_bytes = _aligned(maximum_capacity * descriptor.row_bytes, _COLUMN_ALIGNMENT)
        columns.append(ReplayPhysicalColumn(descriptor, offset, slab_bytes))
        offset += slab_bytes
    return tuple(columns)


def _aligned(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
