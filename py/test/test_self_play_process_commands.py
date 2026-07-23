from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.SelfPlayProcess import SelfPlayProcess
from src.self_play.model_refresh import SearchScheduleState
from src.util.communication import (
    START_CONTINUOUS_SELF_PLAY,
    Communication,
    LATEST_SELF_PLAY_MODEL_VERSION,
    refresh_self_play_model_message,
    self_play_model_refreshed_message,
)


@dataclass
class _FakeSelfPlay:
    dataset: list[int] = field(default_factory=lambda: [1, 2, 3])
    roots: list[int] = field(default_factory=lambda: [10, 11])
    completed_searches: int = 37
    search_schedule_state: SearchScheduleState | None = SearchScheduleState(
        schedule_version=4,
        num_parallel_searches=2,
        num_full_searches=64,
        num_fast_searches=16,
        endgame_shortcut_strength=0.0,
    )
    refreshes: list[tuple[int, Path]] = field(default_factory=list)
    schedule_updates: list[SearchScheduleState] = field(default_factory=list)

    def refresh_model(self, model_version: int, model_path: Path) -> None:
        self.refreshes.append((model_version, model_path))

    def search_schedule(self, schedule_version: int) -> SearchScheduleState:
        return SearchScheduleState(
            schedule_version=schedule_version,
            num_parallel_searches=2,
            num_full_searches=64,
            num_fast_searches=16,
            endgame_shortcut_strength=0.0,
        )

    def update_search_schedule(self, schedule: SearchScheduleState) -> None:
        self.schedule_updates.append(schedule)
        self.search_schedule_state = schedule


@dataclass(frozen=True)
class _FakeCreditTraining:
    maximum_optimizer_steps: int
    optimizer_steps_per_quantum: int


@dataclass(frozen=True)
class _FakeTraining:
    credit_training: _FakeCreditTraining | None


@dataclass(frozen=True)
class _FakeTrainingArguments:
    num_iterations: int
    save_path: str
    training: _FakeTraining = _FakeTraining(None)


def test_pure_refresh_command_preserves_replay_roots_schedule_and_statistics(
    tmp_path: Path,
) -> None:
    process = object.__new__(SelfPlayProcess)
    process.args = _FakeTrainingArguments(num_iterations=12, save_path=str(tmp_path))
    process.node_id = 3
    process.communication = Communication(str(tmp_path / 'communication'))
    process.self_play = _FakeSelfPlay()
    previous_dataset = process.self_play.dataset
    previous_roots = process.self_play.roots
    previous_schedule = process.self_play.search_schedule_state

    process.communication.boardcast(refresh_self_play_model_message(8))
    updated_version = process._refresh_model_if_requested(7)

    assert updated_version == 8
    assert process.self_play.refreshes == [
        (8, tmp_path / 'model_8.jit.pt'),
    ]
    assert process.self_play.dataset is previous_dataset
    assert process.self_play.roots is previous_roots
    assert process.self_play.completed_searches == 37
    assert process.self_play.search_schedule_state is previous_schedule
    assert process.communication.try_receive_from_id(
        self_play_model_refreshed_message(8),
        process.node_id,
    )


def test_credit_mode_continuous_command_has_no_iteration_game_cap(
    tmp_path: Path,
) -> None:
    process = object.__new__(SelfPlayProcess)
    process.args = _FakeTrainingArguments(
        num_iterations=12,
        save_path=str(tmp_path),
        training=_FakeTraining(
            _FakeCreditTraining(
                maximum_optimizer_steps=500_000,
                optimizer_steps_per_quantum=50,
            )
        ),
    )
    process.communication = Communication(str(tmp_path / 'communication'))

    assert process._continuous_self_play_state(7) is None
    process.communication.boardcast(START_CONTINUOUS_SELF_PLAY)

    assert process._continuous_self_play_state(-1) is False
    assert process._continuous_self_play_state(7) is True
    assert process._maximum_model_version() == 10_000


def test_credit_mode_model_refresh_initializes_matching_schedule_first(
    tmp_path: Path,
) -> None:
    process = object.__new__(SelfPlayProcess)
    process.args = _FakeTrainingArguments(
        num_iterations=12,
        save_path=str(tmp_path),
        training=_FakeTraining(
            _FakeCreditTraining(
                maximum_optimizer_steps=500_000,
                optimizer_steps_per_quantum=50,
            )
        ),
    )
    process.node_id = 1
    process.communication = Communication(str(tmp_path / 'communication'))
    process.self_play = _FakeSelfPlay()
    process.self_play.search_schedule_state = None
    process.communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, '9999')

    process._update_search_schedule_if_requested()
    assert process.self_play.search_schedule_state is None
    updated_version = process._refresh_model_if_requested(-1)

    assert updated_version == 9_999
    assert process.self_play.search_schedule_state.schedule_version == 9_999
    assert process.self_play.refreshes == [(9_999, tmp_path / 'model_9999.jit.pt')]


def test_credit_mode_refresh_reads_exact_latest_version_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = object.__new__(SelfPlayProcess)
    process.args = _FakeTrainingArguments(
        num_iterations=12,
        save_path=str(tmp_path),
        training=_FakeTraining(
            _FakeCreditTraining(
                maximum_optimizer_steps=500_000,
                optimizer_steps_per_quantum=50,
            )
        ),
    )
    process.node_id = 2
    process.communication = Communication(str(tmp_path / 'communication'))
    process.self_play = _FakeSelfPlay()
    process.communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, '8765')

    def fail_on_scan(_: str) -> bool:
        raise AssertionError('Credit-mode refresh must not scan version message files.')

    monkeypatch.setattr(process.communication, 'is_received', fail_on_scan)

    assert process._refresh_model_if_requested(8_764) == 8_765
    assert process.self_play.refreshes == [(8_765, tmp_path / 'model_8765.jit.pt')]
    assert process.self_play.search_schedule_state.schedule_version == 8_765


def test_persistent_version_publication_atomically_replaces_complete_value(
    tmp_path: Path,
) -> None:
    communication = Communication(str(tmp_path / 'communication'))

    communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, '7')
    communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, '8765')

    assert communication.try_read(LATEST_SELF_PLAY_MODEL_VERSION) == '8765'
    assert not tuple(communication.folder.glob('.*.tmp'))


def test_credit_mode_rejects_invalid_published_model_version(
    tmp_path: Path,
) -> None:
    process = object.__new__(SelfPlayProcess)
    process.args = _FakeTrainingArguments(
        num_iterations=12,
        save_path=str(tmp_path),
        training=_FakeTraining(
            _FakeCreditTraining(
                maximum_optimizer_steps=500_000,
                optimizer_steps_per_quantum=50,
            )
        ),
    )
    process.communication = Communication(str(tmp_path / 'communication'))
    process.communication.publish_persistent_value(LATEST_SELF_PLAY_MODEL_VERSION, '8partial')

    with pytest.raises(ValueError, match='must be an integer'):
        process._refresh_model_if_requested(7)
