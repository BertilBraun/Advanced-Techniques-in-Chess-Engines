from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType

sys.modules.setdefault('GPUtil', ModuleType('GPUtil'))

from src.cluster.SelfPlayProcess import SelfPlayProcess
from src.self_play.model_refresh import SearchScheduleState
from src.util.communication import (
    Communication,
    refresh_self_play_model_message,
    self_play_model_refreshed_message,
)


@dataclass
class _FakeSelfPlay:
    dataset: list[int] = field(default_factory=lambda: [1, 2, 3])
    roots: list[int] = field(default_factory=lambda: [10, 11])
    completed_searches: int = 37
    search_schedule_state: SearchScheduleState = SearchScheduleState(
        schedule_version=4,
        num_parallel_searches=2,
        num_full_searches=64,
        num_fast_searches=16,
        endgame_shortcut_strength=0.0,
    )
    refreshes: list[tuple[int, Path]] = field(default_factory=list)

    def refresh_model(self, model_version: int, model_path: Path) -> None:
        self.refreshes.append((model_version, model_path))


@dataclass(frozen=True)
class _FakeTrainingArguments:
    num_iterations: int
    save_path: str


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
