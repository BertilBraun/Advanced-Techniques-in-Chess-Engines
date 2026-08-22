from __future__ import annotations

import argparse
import hashlib
import platform
import statistics
import tempfile
import time
from pathlib import Path
from uuid import UUID

import numpy as np
import numpy.typing as npt
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.contracts import WdlTarget
from src.replay.columnar import flatten_column_views
from src.replay.contracts import ReplaySample, SparsePolicyTarget
from src.replay.encoding import encode_replay_columns, encode_replay_rows
from src.replay.layout import ReplayLayout
from src.replay.shard import (
    InboxGameOrder,
    PendingReplayShardManifest,
    ReplayShardGameMetadata,
    ReplayShardReader,
    ReplayShardSourceGame,
    replay_shard_manifest_path,
    write_replay_shard,
)
from src.replay.store import ReplayStore
from src.self_play.completed_game import GameIdentity, SearchVisitCounts, TerminationReason
from src.training.targets import TrainingTargetLayout
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class BoundaryTiming(FrozenModel):
    median_seconds: float
    median_rows_per_second: float
    trial_seconds: tuple[float, ...]


class SyntheticShardBoundaryReport(FrozenModel):
    evidence_scope: str
    comparator_scope: str
    python_version: str
    cpu: str
    games: int
    rows_per_game: int
    total_rows: int
    games_per_shard: int
    shard_count: int
    old_per_game_file_count: int
    columnar_shard_file_count: int
    file_count_reduction: int
    file_count_reduction_factor: float
    repeats: int
    hash_verification_during_boundary: bool
    wrapped_ring: bool
    exact_final_semantics: bool
    semantic_checksum: str
    shard_build_and_write: BoundaryTiming
    old_aos_concatenate: BoundaryTiming
    old_aos_copy_and_flush: BoundaryTiming
    old_aos_boundary_total: BoundaryTiming
    columnar_validation_and_map: BoundaryTiming
    columnar_sequential_append_and_flush: BoundaryTiming
    columnar_boundary_total: BoundaryTiming


def _layout() -> ReplayLayout:
    return ReplayLayout(
        packed_planes=CHESS_STATE_CONTRACT.packed_plane_layout,
        targets=TrainingTargetLayout(
            action_size=CHESS_STATE_CONTRACT.action_size,
            wdl_size=3,
            auxiliary_heads=(),
        ),
        maximum_policy_entries=60,
        maximum_legal_actions=CHESS_STATE_CONTRACT.maximum_legal_action_count,
    )


def _sample(layout: ReplayLayout, row: int) -> ReplaySample:
    legal = tuple((row * 223 + index) % layout.targets.action_size for index in range(32))
    visits = SearchVisitCounts(
        action_ids=legal[:8],
        visit_counts=tuple((row + index) % 31 + 1 for index in range(8)),
    )
    payload = bytes((row + index) % 256 for index in range(layout.packed_planes.payload_bytes))
    return ReplaySample(
        encoded_state=layout.packed_planes.value(payload),
        policy=SparsePolicyTarget(visits=visits, legal_action_ids=legal),
        wdl_target=(
            WdlTarget(win=1.0, draw=0.0, loss=0.0)
            if row % 3 == 0
            else WdlTarget(win=0.0, draw=1.0, loss=0.0)
            if row % 3 == 1
            else WdlTarget(win=0.0, draw=0.0, loss=1.0)
        ),
        root_value=float((row % 21) - 10) / 10.0,
        auxiliary_targets=(),
        sample_weight=float(row % 4 + 1),
        source_model_generation=row,
        source_created_at_seconds=float(1_700_000_000 + row),
    )


def _source(game_number: int) -> ReplayShardSourceGame:
    identity = GameIdentity(
        worker_id=0,
        process_instance_id=UUID('38c8809f-a49d-4d98-8da5-034614893665'),
        game_number=game_number,
    )
    payload = f'synthetic-game-{game_number}'.encode()
    return ReplayShardSourceGame(
        identity=identity,
        order=InboxGameOrder(modified_at_ns=game_number, file_name=identity.file_name),
        source_size=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _shard_inputs(
    layout: ReplayLayout,
    samples: tuple[ReplaySample, ...],
    games: int,
    rows_per_game: int,
    games_per_shard: int,
) -> tuple[tuple[PendingReplayShardManifest, tuple[ReplayShardGameMetadata, ...], tuple[ReplaySample, ...]], ...]:
    sources = tuple(_source(game_number) for game_number in range(games))
    shard_inputs = []
    for sequence, first_game in enumerate(range(0, games, games_per_shard)):
        shard_sources = sources[first_game : first_game + games_per_shard]
        metadata = tuple(
            ReplayShardGameMetadata(
                source=source,
                row_start=index * rows_per_game,
                row_count=rows_per_game,
                length_plies=rows_per_game,
                termination_reason=TerminationReason.NATURAL,
                is_resignation_continuation=False,
                final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
                observations=(),
                policies_truncated=0,
                retained_visit_mass=rows_per_game * 100,
                discarded_visit_mass=0,
            )
            for index, source in enumerate(shard_sources)
        )
        first_row = first_game * rows_per_game
        row_count = len(shard_sources) * rows_per_game
        shard_inputs.append(
            (
                PendingReplayShardManifest.create(layout, sequence, shard_sources),
                metadata,
                samples[first_row : first_row + row_count],
            )
        )
    return tuple(shard_inputs)


def _write_shards(
    staging_path: Path,
    layout: ReplayLayout,
    inputs: tuple[
        tuple[PendingReplayShardManifest, tuple[ReplayShardGameMetadata, ...], tuple[ReplaySample, ...]], ...
    ],
) -> None:
    for pending, metadata, samples in inputs:
        write_replay_shard(staging_path, layout, pending, encode_replay_columns(layout, samples), metadata)


def _seed_store(
    path: Path,
    layout: ReplayLayout,
    total_rows: int,
    initial_samples: tuple[ReplaySample, ...],
) -> ReplayStore:
    maximum_capacity = total_rows + max(1, len(initial_samples) // 2)
    store = ReplayStore.create(path, layout, maximum_capacity=maximum_capacity, logical_capacity=total_rows)
    store.extend(initial_samples)
    store.flush()
    return store


def _columnar_boundary_trial(
    path: Path,
    staging_path: Path,
    layout: ReplayLayout,
    inputs: tuple[
        tuple[PendingReplayShardManifest, tuple[ReplayShardGameMetadata, ...], tuple[ReplaySample, ...]], ...
    ],
    total_rows: int,
    initial_samples: tuple[ReplaySample, ...],
) -> tuple[float, float, float, bool, str]:
    store = _seed_store(path, layout, total_rows, initial_samples)
    readers: list[ReplayShardReader] = []
    try:
        total_started = time.perf_counter()
        validation_started = time.perf_counter()
        for pending, _, _ in inputs:
            readers.append(
                ReplayShardReader.open(
                    replay_shard_manifest_path(staging_path, pending.shard_identity),
                    layout,
                    verify_data_hash=False,
                )
            )
        validation_seconds = time.perf_counter() - validation_started
        append_started = time.perf_counter()
        for reader in readers:
            store.append_columns(reader.columns, reader.manifest.shard_identity)
        store.flush()
        append_seconds = time.perf_counter() - append_started
        total_seconds = time.perf_counter() - total_started
        wrapped = store.state.head + store.state.size > store.state.maximum_capacity
        checksum = _validate_store(store, layout, tuple(sample for _, _, shard in inputs for sample in shard))
        return validation_seconds, append_seconds, total_seconds, wrapped, checksum
    finally:
        for reader in readers:
            reader.close()
        store.close()


def _validate_store(store: ReplayStore, layout: ReplayLayout, expected_samples: tuple[ReplaySample, ...]) -> str:
    gathered = store.gather_logical(np.arange(len(expected_samples), dtype=np.int64))
    expected = encode_replay_columns(layout, expected_samples)
    digest = hashlib.sha256()
    for actual_column, expected_column in zip(
        flatten_column_views(layout, gathered),
        flatten_column_views(layout, expected),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_column.values, expected_column.values)
        digest.update(actual_column.values.tobytes())
    return digest.hexdigest()


def _old_aos_trial(
    path: Path,
    layout: ReplayLayout,
    game_blocks: tuple[npt.NDArray[np.void], ...],
    total_rows: int,
    initial_rows: int,
) -> tuple[float, float, float, str]:
    maximum_capacity = total_rows + max(1, initial_rows // 2)
    ring = np.memmap(path, mode='w+', dtype=layout.row_dtype, shape=(maximum_capacity,))
    total_started = time.perf_counter()
    concatenate_started = time.perf_counter()
    block = np.concatenate(game_blocks)
    concatenate_seconds = time.perf_counter() - concatenate_started
    copy_started = time.perf_counter()
    destination_start = initial_rows % maximum_capacity
    first_count = min(total_rows, maximum_capacity - destination_start)
    ring[destination_start : destination_start + first_count] = block[:first_count]
    if first_count < total_rows:
        ring[: total_rows - first_count] = block[first_count:]
    ring.flush()
    copy_seconds = time.perf_counter() - copy_started
    total_seconds = time.perf_counter() - total_started
    physical = (initial_rows + np.arange(total_rows, dtype=np.int64)) % maximum_capacity
    final_rows = np.asarray(ring[physical])
    np.testing.assert_array_equal(final_rows, block)
    checksum = hashlib.sha256(final_rows.tobytes()).hexdigest()
    del ring
    return concatenate_seconds, copy_seconds, total_seconds, checksum


def _timing(trials: list[float], rows: int) -> BoundaryTiming:
    median = statistics.median(trials)
    return BoundaryTiming(
        median_seconds=median,
        median_rows_per_second=rows / median,
        trial_seconds=tuple(trials),
    )


def run_benchmark(
    output: Path,
    games: int = 256,
    rows_per_game: int = 8,
    games_per_shard: int = 32,
    repeats: int = 5,
) -> SyntheticShardBoundaryReport:
    if games <= 0 or rows_per_game <= 0 or games_per_shard <= 0 or repeats <= 0:
        raise ValueError('Synthetic shard benchmark dimensions and repeats must be positive.')
    if games % games_per_shard:
        raise ValueError('Synthetic games must divide evenly into fixed-size shards.')
    layout = _layout()
    total_rows = games * rows_per_game
    samples = tuple(_sample(layout, row) for row in range(total_rows))
    initial_rows = min(512, max(2, total_rows // 2))
    initial_samples = tuple(_sample(layout, total_rows + row) for row in range(initial_rows))
    inputs = _shard_inputs(layout, samples, games, rows_per_game, games_per_shard)
    game_blocks = tuple(
        encode_replay_rows(layout, samples[start : start + rows_per_game])
        for start in range(0, total_rows, rows_per_game)
    )
    trial_values = {
        name: [] for name in ('shard_write', 'old_concat', 'old_copy', 'old_total', 'validation', 'append', 'total')
    }
    checksums = []
    wrapped = []
    with tempfile.TemporaryDirectory(prefix='az-synthetic-shard-boundary-') as directory:
        root = Path(directory)
        staging_paths = []
        for repeat in range(repeats):
            staging_path = root / f'shards-{repeat}'
            started = time.perf_counter()
            _write_shards(staging_path, layout, inputs)
            trial_values['shard_write'].append(time.perf_counter() - started)
            staging_paths.append(staging_path)
        for repeat in range(repeats):
            old_concat, old_copy, old_total, _ = _old_aos_trial(
                root / f'old-{repeat}.bin', layout, game_blocks, total_rows, initial_rows
            )
            validation, append, total, is_wrapped, checksum = _columnar_boundary_trial(
                root / f'columnar-{repeat}.bin',
                staging_paths[repeat],
                layout,
                inputs,
                total_rows,
                initial_samples,
            )
            trial_values['old_concat'].append(old_concat)
            trial_values['old_copy'].append(old_copy)
            trial_values['old_total'].append(old_total)
            trial_values['validation'].append(validation)
            trial_values['append'].append(append)
            trial_values['total'].append(total)
            wrapped.append(is_wrapped)
            checksums.append(checksum)
    shard_count = len(inputs)
    old_file_count = games * 2
    shard_file_count = shard_count * 2
    report = SyntheticShardBoundaryReport(
        evidence_scope=(
            'Synthetic CPU-only chess-shaped shard/store boundary; not a CUDA, DDP, live-run, quantum-duration, '
            'or <3% ingestion acceptance measurement.'
        ),
        comparator_scope=(
            'AoS reference times preloaded per-game row-block concatenate plus structured memmap copy/flush; '
            'it excludes historical per-file JSON/NPY parsing. Columnar total includes manifest validation/map, '
            'sequential per-shard append, and one store flush; it excludes shard deletion.'
        ),
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        games=games,
        rows_per_game=rows_per_game,
        total_rows=total_rows,
        games_per_shard=games_per_shard,
        shard_count=shard_count,
        old_per_game_file_count=old_file_count,
        columnar_shard_file_count=shard_file_count,
        file_count_reduction=old_file_count - shard_file_count,
        file_count_reduction_factor=old_file_count / shard_file_count,
        repeats=repeats,
        hash_verification_during_boundary=False,
        wrapped_ring=all(wrapped),
        exact_final_semantics=len(set(checksums)) == 1,
        semantic_checksum=checksums[0],
        shard_build_and_write=_timing(trial_values['shard_write'], total_rows),
        old_aos_concatenate=_timing(trial_values['old_concat'], total_rows),
        old_aos_copy_and_flush=_timing(trial_values['old_copy'], total_rows),
        old_aos_boundary_total=_timing(trial_values['old_total'], total_rows),
        columnar_validation_and_map=_timing(trial_values['validation'], total_rows),
        columnar_sequential_append_and_flush=_timing(trial_values['append'], total_rows),
        columnar_boundary_total=_timing(trial_values['total'], total_rows),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(output, report.model_dump_json(indent=2) + '\n')
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the bounded synthetic CPU shard-boundary benchmark.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--games', type=_positive_int, default=256)
    parser.add_argument('--rows-per-game', type=_positive_int, default=8)
    parser.add_argument('--games-per-shard', type=_positive_int, default=32)
    parser.add_argument('--repeats', type=_positive_int, default=5)
    arguments = parser.parse_args()
    run_benchmark(
        arguments.output,
        games=arguments.games,
        rows_per_game=arguments.rows_per_game,
        games_per_shard=arguments.games_per_shard,
        repeats=arguments.repeats,
    )


if __name__ == '__main__':
    main()
