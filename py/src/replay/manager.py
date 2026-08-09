from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Generic, TypeVar

from src.games.contracts import GameStateContract
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.store import ReplayStore
from src.self_play.completed_game import CompletedSelfPlayGame
from src.replay.configuration import ReplayConfiguration
from src.util.frozen_model import FrozenModel


PositionT = TypeVar('PositionT')


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

    @property
    def samples_per_second(self) -> float:
        return self.samples_added / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class ReplayDescription(FrozenModel):
    path: Path
    head: int
    size: int
    logical_capacity: int
    maximum_capacity: int
    layout: ReplayLayout


class ReplayManager(Generic[PositionT]):
    def __init__(
        self,
        inbox_path: Path,
        store: ReplayStore,
        state: GameStateContract[PositionT],
        configuration: ReplayConfiguration,
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

    @classmethod
    def open(
        cls,
        run_path: Path,
        state: GameStateContract[PositionT],
        layout: ReplayLayout,
        configuration: ReplayConfiguration,
        model_generation: int,
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
        return cls(run_path / 'completed-games' / 'inbox', store, state, configuration)

    @property
    def live_samples(self) -> int:
        return self.store.state.size

    def ingest_available_games(self, model_generation: int) -> ReplayIngestion:
        started_at = time.perf_counter()
        before = self.store.state
        self.store.set_logical_capacity(self.configuration.capacity_at(model_generation))
        games_ingested = 0
        samples_added = 0
        policies_truncated = 0
        retained_visit_mass = 0
        discarded_visit_mass = 0
        for path in self._available_games():
            game = CompletedSelfPlayGame.model_validate_json(path.read_text(encoding='utf-8'))
            if path.name != game.identity.file_name:
                raise ValueError(f'Completed-game identity does not match its file name: {path}')
            materialized = materialize_completed_game(
                game,
                self.state,
                self.store.layout.targets,
                self.store.layout.maximum_policy_entries,
            )
            for sample in materialized.samples:
                self.store.append(sample)
            path.unlink()
            games_ingested += 1
            samples_added += len(materialized.samples)
            policies_truncated += materialized.policies_truncated
            retained_visit_mass += materialized.retained_visit_mass
            discarded_visit_mass += materialized.discarded_visit_mass
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
