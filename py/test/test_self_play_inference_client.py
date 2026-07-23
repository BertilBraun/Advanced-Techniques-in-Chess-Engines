import copy
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import pytest

from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame, has_positive_visit_counts
from src.settings import TRAINING_ARGS
from src.train.TrainingArgs import DirectSelfPlayParams


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
class _FakeInferenceClientParams:
    device_id: int
    currentModelPath: str
    maxBatchSize: int
    microsecondsTimeoutInferenceThread: int
    cacheCapacity: int


@dataclass(frozen=True)
class _FakeMctsParams:
    num_parallel_searches: int
    num_full_searches: int
    num_fast_searches: int
    dirichlet_alpha: float
    dirichlet_epsilon: float
    c_param: float
    min_visit_count: int
    num_threads: int


@dataclass(frozen=True)
class _FakeDirectSelfPlayInferenceParams:
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int


class _FakeMcts:
    use_inference_cache: bool | None = None
    direct_inference_params: _FakeDirectSelfPlayInferenceParams | None = None

    def __init__(
        self,
        client_args: _FakeInferenceClientParams,
        mcts_args: _FakeMctsParams,
        use_inference_cache: bool,
        direct_inference_params: _FakeDirectSelfPlayInferenceParams | None,
        initial_model_version: int,
    ) -> None:
        assert client_args.cacheCapacity == 0
        assert mcts_args.num_full_searches > mcts_args.num_parallel_searches
        _FakeMcts.use_inference_cache = use_inference_cache
        _FakeMcts.direct_inference_params = direct_inference_params
        assert initial_model_version >= 0
        self.arena_capacity = (
            max(mcts_args.num_full_searches, mcts_args.num_fast_searches) + mcts_args.num_parallel_searches + 1
        )


class _LifecycleMcts:
    def __init__(self, events: list[str], arena_capacity: int) -> None:
        self.events = events
        self.arena_capacity = arena_capacity

    def get_inference_statistics(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        inference_statistics = SimpleNamespace(
            cacheHitRate=0.0,
            uniquePositions=0,
            cacheSizeMB=0,
            cacheCapacity=0,
            cacheEvictions=0,
            cacheFingerprintCollisions=0,
            nnOutputValueDistribution=[],
            averageNumberOfPositionsInInferenceCall=0.0,
        )
        return inference_statistics, SimpleNamespace(functionTimes=[])

    def refresh_model(self, model_version: int, model_path: str) -> None:
        self.events.append(f'refresh:{model_version}:{model_path}')

    def update_search_schedule(self, parameters: _FakeMctsParams) -> bool:
        previous_capacity = self.arena_capacity
        self.events.append(f'schedule:{parameters.num_full_searches}')
        self.arena_capacity = (
            max(parameters.num_full_searches, parameters.num_fast_searches) + parameters.num_parallel_searches + 1
        )
        return previous_capacity != self.arena_capacity


class _LifecycleRoot:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __del__(self) -> None:
        self.events.append('release_root')


def test_self_play_constructs_selected_inference_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParams
    fake_alpha_zero_cpp.DirectSelfPlayInferenceParams = _FakeDirectSelfPlayInferenceParams
    fake_alpha_zero_cpp.MCTS = _FakeMcts
    fake_alpha_zero_cpp.MCTSParams = _FakeMctsParams
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    training_args = copy.deepcopy(TRAINING_ARGS)
    training_args.self_play.use_inference_cache = False
    training_args.self_play.inference_cache_capacity = 0
    self_play = SelfPlayCpp(device_id=0, args=training_args)

    self_play._set_mcts(iteration=50)

    assert _FakeMcts.use_inference_cache is False
    assert _FakeMcts.direct_inference_params is None


def test_self_play_constructs_direct_inference_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParams
    fake_alpha_zero_cpp.DirectSelfPlayInferenceParams = _FakeDirectSelfPlayInferenceParams
    fake_alpha_zero_cpp.MCTS = _FakeMcts
    fake_alpha_zero_cpp.MCTSParams = _FakeMctsParams
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    training_args = copy.deepcopy(TRAINING_ARGS)
    training_args.self_play.use_inference_cache = False
    training_args.self_play.inference_cache_capacity = 0
    training_args.self_play.direct_inference = DirectSelfPlayParams(
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )
    self_play = SelfPlayCpp(device_id=0, args=training_args)

    self_play._set_mcts(iteration=50)

    assert _FakeMcts.direct_inference_params == _FakeDirectSelfPlayInferenceParams(2, 64, 1)


def test_model_refresh_retains_self_play_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    self_play = object.__new__(SelfPlayCpp)
    self_play.args = copy.deepcopy(TRAINING_ARGS.self_play)
    self_play.dataset = [object()]
    self_play.iteration = 0
    self_play.model_version = 0
    self_play.model_refresh_acknowledgements = [0]
    self_play.search_schedule_state = self_play.search_schedule(0)
    root = _LifecycleRoot(events)
    game = SimpleNamespace(already_expanded_node=root)
    self_play.self_play_games = [game]
    self_play.completed_searches = 19
    self_play.mcts = _LifecycleMcts(events, self_play.search_schedule_state.arena_capacity)
    previous_mcts = self_play.mcts
    monkeypatch.setattr('src.self_play.SelfPlayCpp.log_scalar', lambda *_args: None)

    self_play.refresh_model(1, 'updated.jit.pt')

    assert self_play.mcts is previous_mcts
    assert game.already_expanded_node is root
    assert len(self_play.dataset) == 1
    assert self_play.completed_searches == 19
    assert self_play.search_schedule_state == self_play.search_schedule(0)
    assert self_play.model_version == 1
    assert self_play.model_refresh_acknowledgements == [0, 1]
    assert events == ['refresh:1:updated.jit.pt']


def test_failed_model_refresh_is_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    self_play = object.__new__(SelfPlayCpp)
    self_play.args = copy.deepcopy(TRAINING_ARGS.self_play)
    self_play.iteration = 0
    self_play.model_version = 7
    self_play.model_refresh_acknowledgements = [7]
    self_play.search_schedule_state = self_play.search_schedule(0)
    self_play.self_play_games = [SimpleNamespace(already_expanded_node=object())]
    self_play.mcts = _LifecycleMcts([], self_play.search_schedule_state.arena_capacity)

    def fail_refresh(_model_version: int, _model_path: str) -> None:
        raise RuntimeError('invalid checkpoint')

    monkeypatch.setattr(self_play.mcts, 'refresh_model', fail_refresh)

    with pytest.raises(RuntimeError, match='invalid checkpoint'):
        self_play.refresh_model(8, 'broken.jit.pt')

    assert self_play.model_version == 7
    assert self_play.model_refresh_acknowledgements == [7]


def test_diagnostic_refresh_can_discard_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    self_play = object.__new__(SelfPlayCpp)
    self_play.args = copy.deepcopy(TRAINING_ARGS.self_play)
    self_play.iteration = 0
    self_play.model_version = 0
    self_play.model_refresh_acknowledgements = [0]
    self_play.search_schedule_state = self_play.search_schedule(0)
    self_play.self_play_games = [SimpleNamespace(already_expanded_node=object())]
    self_play.mcts = _LifecycleMcts([], self_play.search_schedule_state.arena_capacity)
    monkeypatch.setattr('src.self_play.SelfPlayCpp.log_scalar', lambda *_args: None)

    self_play.refresh_model(1, 'updated.jit.pt', discard_roots=True)

    assert self_play.self_play_games[0].already_expanded_node is None
