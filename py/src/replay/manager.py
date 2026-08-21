from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

import numpy as np
from pydantic import Field
from src.experiment.configuration import ExperimentConfiguration
from src.experiment.generation_schedule import FloatGenerationSchedule
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.configuration import ReplayConfiguration
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.parallel_materialization import (
    STAGED_ROWS_SUFFIX,
    StagedGame,
    StagedGameMetadata,
    initialize_materialization_worker,
    stage_completed_game,
    stage_completed_game_path,
    staged_metadata_path,
    staged_rows_path,
)
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchObservation, TerminationReason
from src.self_play.resignation import ResignationCalibrator
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.log import log

PositionT = TypeVar('PositionT')

DISPATCH_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class IngestedCompletedGame:
    length_plies: int
    termination_reason: TerminationReason
    observations: tuple[SearchObservation, ...] = ()


@dataclass(frozen=True)
class ReplayIngestion:
    games_ingested: int
    samples_added: int
    live_samples: int
    evicted_samples: int
    policies_truncated: int
    retained_visit_mass: int
    discarded_visit_mass: int
    elapsed_seconds: float
    completed_games: tuple[IngestedCompletedGame, ...]

    @property
    def samples_per_second(self) -> float:
        return self.samples_added / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class StagedAppendManifest(FrozenModel):
    schema_version: Literal[1] = 1
    game_ids: tuple[str, ...]
    total_rows_after_append: int = Field(ge=0)


@dataclass(frozen=True)
class _CompleteStagedGame:
    game_id: str
    rows_path: Path
    metadata_path: Path


class _MaterializationDispatcher:
    def __init__(self, manager: ReplayManager[PositionT]) -> None:
        self._manager = manager
        self.on_staged: Callable[[StagedGame], None] = lambda staged: None
        self.poll_interval_seconds = DISPATCH_INTERVAL_SECONDS
        self._claimed: set[str] = set()
        self._pending: dict[Future[StagedGame], str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.failed_game_count = 0

    def start(self) -> None:
        assert self._thread is None
        self._thread = threading.Thread(target=self._run, name='replay-materialization-dispatcher', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._pending:
            wait(tuple(self._pending))
            self._collect_finished()

    def dispatch_once(self) -> None:
        self._collect_finished()
        inbox_files = self._manager.inbox_files_by_modification_time()
        self._claimed &= {inbox_file.stem for inbox_file in inbox_files} | set(self._pending.values())
        for inbox_file in inbox_files:
            game_id = inbox_file.stem
            if game_id in self._claimed:
                continue
            if self._manager.is_staged(game_id):
                # Left over from a crash between staging and the worker's inbox unlink.
                inbox_file.unlink(missing_ok=True)
                continue
            self._claimed.add(game_id)
            executor = self._manager.materialization_executor
            if executor is not None:
                future = executor.submit(stage_completed_game_path, inbox_file, self._manager.staging_path)
                self._pending[future] = game_id
            else:
                self._stage_inline(inbox_file, game_id)
        self._collect_finished()

    def drain(self) -> None:
        while True:
            self.dispatch_once()
            if not self._pending and not self._has_unclaimed_inbox_files():
                return
            time.sleep(0.01)

    def _has_unclaimed_inbox_files(self) -> bool:
        return any(
            inbox_file.stem not in self._claimed for inbox_file in self._manager.inbox_files_by_modification_time()
        )

    def _stage_inline(self, inbox_file: Path, game_id: str) -> None:
        try:
            staged = self._manager.stage_game_inline(inbox_file)
        except Exception as error:  # noqa: BLE001 - a bad game must not stop the pipeline
            self.failed_game_count += 1
            log(f'Failed to materialize completed game {game_id}: {error}')
            return
        self.on_staged(staged)

    def _collect_finished(self) -> None:
        for future in [pending for pending in self._pending if pending.done()]:
            game_id = self._pending.pop(future)
            error = future.exception()
            if error is not None:
                self.failed_game_count += 1
                log(f'Failed to materialize completed game {game_id}: {error}')
                continue
            self.on_staged(future.result())

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self.dispatch_once()


class ReplayManager(Generic[PositionT]):
    def __init__(
        self,
        completed_games_path: Path,
        store: ReplayStore,
        state: GameStateContract[PositionT],
        configuration: ReplayConfiguration,
        value_discount_per_ply: FloatGenerationSchedule,
        resignation_calibrator: ResignationCalibrator | None,
        terminal_oracle: TerminalOracle[PositionT] | None,
        experiment_configuration: ExperimentConfiguration | None,
    ) -> None:
        if store.layout.packed_planes != state.packed_plane_layout:
            raise ValueError('Replay layout does not match the game packed-plane representation.')
        if store.layout.targets.action_size != state.action_size:
            raise ValueError('Replay layout does not match the game action count.')
        if store.layout.maximum_policy_entries != configuration.maximum_policy_entries:
            raise ValueError('Replay layout does not match replay policy retention configuration.')
        if store.state.maximum_capacity != configuration.maximum_capacity:
            raise ValueError('Replay file does not match replay maximum capacity configuration.')
        self.inbox_path = completed_games_path / 'inbox'
        self.staging_path = completed_games_path / 'staging'
        self.append_manifest_path = completed_games_path / 'last-append.json'
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        self.staging_path.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.state = state
        self.configuration = configuration
        self.value_discount_per_ply = value_discount_per_ply
        self.resignation_calibrator = resignation_calibrator
        self.terminal_oracle = terminal_oracle
        if configuration.materialization_processes > 1 and experiment_configuration is None:
            raise ValueError('Parallel replay materialization requires the experiment configuration.')
        self.materialization_executor = (
            None
            if configuration.materialization_processes == 1
            else ProcessPoolExecutor(
                max_workers=configuration.materialization_processes,
                initializer=initialize_materialization_worker,
                initargs=(experiment_configuration.model_dump_json(), configuration.maximum_policy_entries),
            )
        )
        self._dispatcher = _MaterializationDispatcher(self)
        self._recover_directories()

    @classmethod
    def open(
        cls,
        run_path: Path,
        state: GameStateContract[PositionT],
        layout: ReplayLayout,
        configuration: ReplayConfiguration,
        model_generation: int,
        value_discount_per_ply: FloatGenerationSchedule,
        terminal_oracle: TerminalOracle[PositionT] | None,
        resignation_calibrator: ResignationCalibrator | None = None,
        experiment_configuration: ExperimentConfiguration | None = None,
    ) -> ReplayManager[PositionT]:
        replay_path = run_path / 'replay.bin'
        if replay_path.exists():
            store = ReplayStore.open(replay_path, layout)
        else:
            store = ReplayStore.create(
                replay_path,
                layout,
                configuration.maximum_capacity,
                configuration.capacity_at(model_generation),
            )
        return cls(
            run_path / 'completed-games',
            store,
            state,
            configuration,
            value_discount_per_ply,
            resignation_calibrator,
            terminal_oracle,
            experiment_configuration,
        )

    @property
    def live_samples(self) -> int:
        return self.store.state.size

    @property
    def inbox_depth(self) -> int:
        return len(self.inbox_files_by_modification_time())

    @property
    def staging_depth(self) -> int:
        return len(self._complete_staged_games())

    @property
    def materialization_failures(self) -> int:
        return self._dispatcher.failed_game_count

    def total_materialized_samples(self) -> int:
        staged_rows = 0
        for staged in self._complete_staged_games():
            metadata = StagedGameMetadata.model_validate_json(staged.metadata_path.read_text(encoding='utf-8'))
            staged_rows += metadata.row_count
        return self.store.total_appended_rows + staged_rows

    def start_materialization(
        self,
        on_staged: Callable[[StagedGame], None],
        poll_interval_seconds: float = DISPATCH_INTERVAL_SECONDS,
    ) -> None:
        self._dispatcher.on_staged = on_staged
        self._dispatcher.poll_interval_seconds = poll_interval_seconds
        self._dispatcher.start()

    def materialize_available_games(self, on_staged: Callable[[StagedGame], None]) -> None:
        self._dispatcher.on_staged = on_staged
        self._dispatcher.drain()

    def append_staged_games(self, model_generation: int) -> ReplayIngestion:
        started_at = time.perf_counter()
        before = self.store.state
        self.store.set_logical_capacity(self.configuration.capacity_at(model_generation))
        staged_games = self._complete_staged_games()
        if not staged_games:
            after = self.store.state
            return ReplayIngestion(
                games_ingested=0,
                samples_added=0,
                live_samples=after.size,
                evicted_samples=after.evicted_rows - before.evicted_rows,
                policies_truncated=0,
                retained_visit_mass=0,
                discarded_visit_mass=0,
                elapsed_seconds=time.perf_counter() - started_at,
                completed_games=(),
            )
        blocks: list[np.ndarray] = []
        metadata_by_game: list[StagedGameMetadata] = []
        for staged in staged_games:
            metadata = StagedGameMetadata.model_validate_json(staged.metadata_path.read_text(encoding='utf-8'))
            rows = np.load(staged.rows_path, allow_pickle=False)
            if rows.dtype != self.store.layout.row_dtype:
                raise ValueError(f'Staged replay rows do not match the store layout: {staged.rows_path}')
            if len(rows) != metadata.row_count:
                raise ValueError(f'Staged replay rows disagree with their metadata row count: {staged.rows_path}')
            blocks.append(rows)
            metadata_by_game.append(metadata)
        block = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        manifest = StagedAppendManifest(
            game_ids=tuple(staged.game_id for staged in staged_games),
            total_rows_after_append=self.store.total_appended_rows + len(block),
        )
        write_text_atomically(self.append_manifest_path, manifest.model_dump_json() + '\n')
        self.store.extend_rows(block)
        self.store.flush()
        self._observe_resignation_games(metadata_by_game)
        self._unlink_staged_games(staged_games)
        self.append_manifest_path.unlink(missing_ok=True)
        after = self.store.state
        return ReplayIngestion(
            games_ingested=len(staged_games),
            samples_added=len(block),
            live_samples=after.size,
            evicted_samples=after.evicted_rows - before.evicted_rows,
            policies_truncated=sum(metadata.policies_truncated for metadata in metadata_by_game),
            retained_visit_mass=sum(metadata.retained_visit_mass for metadata in metadata_by_game),
            discarded_visit_mass=sum(metadata.discarded_visit_mass for metadata in metadata_by_game),
            elapsed_seconds=time.perf_counter() - started_at,
            completed_games=tuple(
                IngestedCompletedGame(
                    length_plies=metadata.length_plies,
                    termination_reason=metadata.termination_reason,
                    observations=metadata.observations,
                )
                for metadata in metadata_by_game
            ),
        )

    def description(self) -> ReplayDescription:
        state = self.store.state
        return ReplayDescription(
            path=self.store.path,
            head=state.head,
            size=state.size,
            logical_capacity=state.logical_capacity,
            maximum_capacity=state.maximum_capacity,
            layout=self.store.layout,
        )

    def close(self) -> None:
        self._dispatcher.stop()
        if self.materialization_executor is not None:
            self.materialization_executor.shutdown()
        self.store.close()

    def is_staged(self, game_id: str) -> bool:
        return (
            staged_metadata_path(self.staging_path, game_id).exists()
            and staged_rows_path(self.staging_path, game_id).exists()
        )

    def inbox_files_by_modification_time(self) -> tuple[Path, ...]:
        return _files_by_modification_time(self.inbox_path, '*.json')

    def stage_game_inline(self, inbox_file: Path) -> StagedGame:
        return stage_completed_game(
            inbox_file,
            self.staging_path,
            self.state,
            self.terminal_oracle,
            self.store.layout,
            self.value_discount_per_ply,
        )

    def _complete_staged_games(self) -> tuple[_CompleteStagedGame, ...]:
        complete = []
        for rows_path in _files_by_modification_time(self.staging_path, f'*{STAGED_ROWS_SUFFIX}'):
            game_id = rows_path.name.removesuffix(STAGED_ROWS_SUFFIX)
            metadata_path = staged_metadata_path(self.staging_path, game_id)
            if metadata_path.exists():
                complete.append(_CompleteStagedGame(game_id=game_id, rows_path=rows_path, metadata_path=metadata_path))
        return tuple(complete)

    def _observe_resignation_games(self, metadata_by_game: list[StagedGameMetadata]) -> None:
        if self.resignation_calibrator is None:
            return
        with self.resignation_calibrator.calibration_batch() as calibration_batch:
            for metadata in metadata_by_game:
                calibration_batch.observe_game_record(
                    archive_key=metadata.identity.archive_key,
                    is_resignation_continuation=metadata.is_resignation_continuation,
                    termination_reason=metadata.termination_reason,
                    final_wdl=metadata.final_wdl,
                    length_plies=metadata.length_plies,
                    observations=metadata.observations,
                )

    def _unlink_staged_games(self, staged_games: tuple[_CompleteStagedGame, ...]) -> None:
        for staged in staged_games:
            staged.rows_path.unlink(missing_ok=True)
            staged.metadata_path.unlink(missing_ok=True)

    def _recover_directories(self) -> None:
        for directory in (self.inbox_path, self.staging_path):
            for orphaned_temporary_file in directory.glob('.*.tmp'):
                orphaned_temporary_file.unlink(missing_ok=True)
        if self.append_manifest_path.exists():
            manifest = StagedAppendManifest.model_validate_json(self.append_manifest_path.read_text(encoding='utf-8'))
            if self.store.total_appended_rows >= manifest.total_rows_after_append:
                # The append completed before the crash; only the staged-file cleanup is outstanding.
                for game_id in manifest.game_ids:
                    staged_rows_path(self.staging_path, game_id).unlink(missing_ok=True)
                    staged_metadata_path(self.staging_path, game_id).unlink(missing_ok=True)
            self.append_manifest_path.unlink(missing_ok=True)
        for inbox_file in self.inbox_files_by_modification_time():
            if self.is_staged(inbox_file.stem):
                inbox_file.unlink(missing_ok=True)


def _files_by_modification_time(directory: Path, pattern: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    # Materialization workers unlink files concurrently; a vanished file is simply dropped from the listing.
    modified_at_by_path: dict[Path, float] = {}
    for path in directory.glob(pattern):
        try:
            modified_at_by_path[path] = path.stat().st_mtime
        except OSError:
            continue
    return tuple(sorted(modified_at_by_path, key=lambda path: (modified_at_by_path[path], path.name)))
