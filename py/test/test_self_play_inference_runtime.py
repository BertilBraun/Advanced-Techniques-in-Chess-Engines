import copy
from pathlib import Path
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import pytest

from src.games.chess.self_play import ChessSelfPlayPolicy, SelfPlayGame, has_positive_visit_counts
from src.self_play.completed_game import CompletedGamePublisher
from src.games.chess.search_schedule import SearchScheduleState
from test_helpers.chess_configuration import CHESS_SELF_PLAY
from src.training.configuration import BatchedInferenceParams


@pytest.mark.parametrize(
    ('visit_counts', 'expected'),
    (
        ([], False),
        ([(1, 0), (2, 0)], False),
        ([(1, -1), (2, 2)], False),
        ([(1, 1), (2, 0)], True),
    ),
)
def test_positive_visit_count_validation(visit_counts: list[tuple[int, int]], expected: bool) -> None:
    assert has_positive_visit_counts(visit_counts) is expected


def test_game_tracks_model_version_range_across_copies() -> None:
    game = SelfPlayGame()
    game.acknowledge_model_version(4)
    copied_game = game.copy()
    copied_game.acknowledge_model_version(7)

    assert copied_game.oldest_model_version == 4
    assert copied_game.newest_model_version == 7
    assert game.oldest_model_version == 4
    assert game.newest_model_version == 4


@dataclass(frozen=True)
class _FakeInferenceConfiguration:
    device_id: int
    model_path: str


@dataclass(frozen=True)
class _FakeSelfPlaySearchParameters:
    parallel_searches: int
    full_searches: int
    fast_searches: int
    dirichlet_alpha: float
    dirichlet_epsilon: float
    exploration_constant: float
    minimum_root_visits: int


@dataclass(frozen=True)
class _FakeBatchedInferenceParameters:
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int


class _FakeChessSelfPlaySearch:
    inference_parameters: _FakeBatchedInferenceParameters | None = None

    def __init__(
        self,
        runtime_parameters: _FakeInferenceConfiguration,
        search_parameters: _FakeSelfPlaySearchParameters,
        inference_parameters: _FakeBatchedInferenceParameters | None,
        initial_model_version: int,
    ) -> None:
        assert runtime_parameters.model_path.endswith('.jit.pt')
        assert search_parameters.full_searches > search_parameters.parallel_searches
        _FakeChessSelfPlaySearch.inference_parameters = inference_parameters
        assert initial_model_version >= 0
        self.arena_capacity = (
            max(search_parameters.full_searches, search_parameters.fast_searches)
            + search_parameters.parallel_searches
            + 1
        )


class _LifecycleSearch:
    def __init__(self, events: list[str], arena_capacity: int) -> None:
        self.events = events
        self.arena_capacity = arena_capacity

    def inference_statistics(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        inference_statistics = SimpleNamespace(
            averageNumberOfPositionsInInferenceCall=0.0,
        )
        return inference_statistics, SimpleNamespace(functionTimes=[])

    def refresh_model(self, model_version: int, model_path: str) -> None:
        self.events.append(f'refresh:{model_version}:{model_path}')

    def update_search_schedule(self, parameters: _FakeSelfPlaySearchParameters) -> bool:
        previous_capacity = self.arena_capacity
        self.events.append(f'schedule:{parameters.full_searches}')
        self.arena_capacity = max(parameters.full_searches, parameters.fast_searches) + parameters.parallel_searches + 1
        return previous_capacity != self.arena_capacity


class _LifecycleRoot:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self) -> None:
        self.events.append('reset_root')

    def __del__(self) -> None:
        self.events.append('release_root')


def _fake_search_parameters(schedule: SearchScheduleState) -> _FakeSelfPlaySearchParameters:
    return _FakeSelfPlaySearchParameters(
        parallel_searches=schedule.num_parallel_searches,
        full_searches=schedule.num_full_searches,
        fast_searches=schedule.num_fast_searches,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        exploration_constant=1.0,
        minimum_root_visits=0,
    )


def test_self_play_constructs_batched_inference_runtime_during_search_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceConfiguration = _FakeInferenceConfiguration
    fake_alpha_zero_cpp.BatchedInferenceParameters = _FakeBatchedInferenceParameters
    fake_alpha_zero_cpp.ChessSelfPlaySearch = _FakeChessSelfPlaySearch
    fake_alpha_zero_cpp.SelfPlaySearchParameters = _FakeSelfPlaySearchParameters
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    search = CHESS_SELF_PLAY.search.validated_copy(
        update={
            'full_searches': {
                'kind': 'linear',
                'start_generation': 0,
                'end_generation': 100,
                'start_value': 100,
                'end_value': 600,
                'rounding': 'nearest',
            },
            'fast_searches': {'kind': 'constant', 'value': 25},
        }
    )
    self_play_args = CHESS_SELF_PLAY.validated_copy(update={'search': search.model_dump(mode='json')})
    self_play = ChessSelfPlayPolicy(
        device_id=0,
        configuration=self_play_args,
        save_path=str(tmp_path),
        completed_game_publisher=CompletedGamePublisher(tmp_path, 0, 0),
    )

    self_play.refresh_model(50, tmp_path / 'model-50.jit.pt', ())

    assert _FakeChessSelfPlaySearch.inference_parameters == _FakeBatchedInferenceParameters(2, 64, 2)
    assert self_play.search_schedule(0).num_full_searches == 100
    assert self_play.search_schedule(1).num_full_searches == 105
    assert self_play.search_schedule(100).num_full_searches == 600


def test_self_play_constructs_direct_inference_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceConfiguration = _FakeInferenceConfiguration
    fake_alpha_zero_cpp.BatchedInferenceParameters = _FakeBatchedInferenceParameters
    fake_alpha_zero_cpp.ChessSelfPlaySearch = _FakeChessSelfPlaySearch
    fake_alpha_zero_cpp.SelfPlaySearchParameters = _FakeSelfPlaySearchParameters
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    batched_inference = BatchedInferenceParams(
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )
    self_play_args = CHESS_SELF_PLAY.validated_copy(update={'inference': batched_inference.model_dump(mode='json')})
    self_play = ChessSelfPlayPolicy(
        device_id=0,
        configuration=self_play_args,
        save_path=str(tmp_path),
        completed_game_publisher=CompletedGamePublisher(tmp_path, 0, 0),
    )

    self_play.refresh_model(50, tmp_path / 'model-50.jit.pt', ())

    assert _FakeChessSelfPlaySearch.inference_parameters == _FakeBatchedInferenceParameters(2, 64, 1)


def test_model_refresh_retains_game_state_and_resets_search_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    self_play = object.__new__(ChessSelfPlayPolicy)
    self_play.args = copy.deepcopy(CHESS_SELF_PLAY)
    self_play.resolved_parameters = self_play._resolve_parameters(0)
    self_play.dataset = [object()]
    self_play.iteration = 0
    self_play.model_version = 0
    self_play.model_refresh_acknowledgements = [0]
    self_play.search_schedule_state = self_play.search_schedule(0)
    root = _LifecycleRoot(events)
    game = SimpleNamespace(already_expanded_node=root)
    self_play.completed_searches = 19
    self_play.search_engine = _LifecycleSearch(events, self_play.search_schedule_state.arena_capacity)
    previous_search = self_play.search_engine
    monkeypatch.setattr('src.games.chess.self_play.log_scalar', lambda *_args: None)
    monkeypatch.setattr(self_play, '_native_mcts_params', _fake_search_parameters)

    self_play.refresh_model(1, Path('updated.jit.pt'), (game,))

    assert self_play.search_engine is previous_search
    assert game.already_expanded_node is root
    assert len(self_play.dataset) == 1
    assert self_play.completed_searches == 19
    assert self_play.search_schedule_state == self_play.search_schedule(1)
    assert self_play.model_version == 1
    assert self_play.model_refresh_acknowledgements == [0, 1]
    assert events == [
        f'schedule:{self_play.search_schedule(1).num_full_searches}',
        'refresh:1:updated.jit.pt',
        'reset_root',
    ]


def test_failed_model_refresh_is_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    self_play = object.__new__(ChessSelfPlayPolicy)
    self_play.args = copy.deepcopy(CHESS_SELF_PLAY)
    self_play.resolved_parameters = self_play._resolve_parameters(0)
    self_play.iteration = 0
    self_play.model_version = 7
    self_play.model_refresh_acknowledgements = [7]
    self_play.search_schedule_state = self_play.search_schedule(0)
    game = SimpleNamespace(already_expanded_node=object())
    self_play.search_engine = _LifecycleSearch([], self_play.search_schedule_state.arena_capacity)
    monkeypatch.setattr(self_play, '_native_mcts_params', _fake_search_parameters)

    def fail_refresh(_model_version: int, _model_path: str) -> None:
        raise RuntimeError('invalid checkpoint')

    monkeypatch.setattr(self_play.search_engine, 'refresh_model', fail_refresh)

    with pytest.raises(RuntimeError, match='invalid checkpoint'):
        self_play.refresh_model(8, Path('broken.jit.pt'), (game,))

    assert self_play.model_version == 7
    assert self_play.model_refresh_acknowledgements == [7]
