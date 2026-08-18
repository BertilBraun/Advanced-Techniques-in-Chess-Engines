from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from src.experiment.generation_schedule import FloatGenerationSchedule
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.configuration import ReplayConfiguration
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.store import ReplayStore
from src.self_play.completed_game import CompletedSelfPlayGame, SearchObservation, TerminationReason
from src.self_play.resignation import ResignationCalibrationBatch, ResignationCalibrator

PositionT = TypeVar('PositionT')


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


class ReplayManager(Generic[PositionT]):
    def __init__(
        self,
        inbox_path: Path,
        store: ReplayStore,
        state: GameStateContract[PositionT],
        configuration: ReplayConfiguration,
        value_discount_per_ply: FloatGenerationSchedule,
        resignation_calibrator: ResignationCalibrator | None,
        terminal_oracle: TerminalOracle[PositionT] | None,
    ) -> None:
        if store.layout.packed_planes != state.packed_plane_layout:
            raise ValueError('Replay layout does not match the game packed-plane representation.')
        if store.layout.targets.action_size != state.action_size:
            raise ValueError('Replay layout does not match the game action count.')
        if store.layout.maximum_policy_entries != configuration.maximum_policy_entries:
            raise ValueError('Replay layout does not match replay policy retention configuration.')
        if store.state.maximum_capacity != configuration.maximum_capacity:
            raise ValueError('Replay file does not match replay maximum capacity configuration.')
        self.inbox_path = inbox_path
        self.store = store
        self.state = state
        self.configuration = configuration
        self.value_discount_per_ply = value_discount_per_ply
        self.resignation_calibrator = resignation_calibrator
        self.terminal_oracle = terminal_oracle

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
            run_path / 'completed-games' / 'inbox',
            store,
            state,
            configuration,
            value_discount_per_ply,
            resignation_calibrator,
            terminal_oracle,
        )

    @property
    def live_samples(self) -> int:
        return self.store.state.size

    def ingest_available_games(self, model_generation: int) -> ReplayIngestion:
        if self.resignation_calibrator is None:
            return self._ingest_available_games(model_generation, None)
        with self.resignation_calibrator.calibration_batch() as calibration_batch:
            return self._ingest_available_games(model_generation, calibration_batch)

    def _ingest_available_games(
        self,
        model_generation: int,
        calibration_batch: ResignationCalibrationBatch | None,
    ) -> ReplayIngestion:
        started_at = time.perf_counter()
        before = self.store.state
        self.store.set_logical_capacity(self.configuration.capacity_at(model_generation))
        games_ingested = 0
        samples_added = 0
        policies_truncated = 0
        retained_visit_mass = 0
        discarded_visit_mass = 0
        completed_games: list[IngestedCompletedGame] = []
        for path in self._available_games():
            game = CompletedSelfPlayGame.model_validate_json(path.read_text(encoding='utf-8'))
            if path.name != game.identity.file_name:
                raise ValueError(f'Completed-game identity does not match its file name: {path}')
            materialized = materialize_completed_game(
                game,
                self.state,
                self.terminal_oracle,
                self.store.layout.targets,
                self.store.layout.maximum_policy_entries,
                self.value_discount_per_ply,
            )
            for sample in materialized.samples:
                self.store.append(sample)
            if calibration_batch is not None:
                calibration_batch.observe_completed_game(game)
            path.unlink()
            games_ingested += 1
            samples_added += len(materialized.samples)
            policies_truncated += materialized.policies_truncated
            retained_visit_mass += materialized.retained_visit_mass
            discarded_visit_mass += materialized.discarded_visit_mass
            completed_games.append(
                IngestedCompletedGame(
                    length_plies=len(game.action_ids),
                    termination_reason=game.termination_reason,
                    observations=game.observations,
                )
            )
        self.store.flush()
        after = self.store.state
        return ReplayIngestion(
            games_ingested=games_ingested,
            samples_added=samples_added,
            live_samples=after.size,
            evicted_samples=after.evicted_rows - before.evicted_rows,
            policies_truncated=policies_truncated,
            retained_visit_mass=retained_visit_mass,
            discarded_visit_mass=discarded_visit_mass,
            elapsed_seconds=time.perf_counter() - started_at,
            completed_games=tuple(completed_games),
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
        self.store.close()

    def _available_games(self) -> tuple[Path, ...]:
        if not self.inbox_path.exists():
            return ()
        return tuple(sorted(self.inbox_path.glob('*.json')))
