import copy
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import pytest

from src.self_play.SelfPlayCpp import SelfPlayCpp
from src.settings import TRAINING_ARGS


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


class _FakeMcts:
    use_inference_cache: bool | None = None

    def __init__(
        self,
        client_args: _FakeInferenceClientParams,
        mcts_args: _FakeMctsParams,
        use_inference_cache: bool,
    ) -> None:
        assert client_args.cacheCapacity == 0
        assert mcts_args.num_full_searches > mcts_args.num_parallel_searches
        _FakeMcts.use_inference_cache = use_inference_cache


class _LifecycleMcts:
    def __init__(self, events: list[str]) -> None:
        self.events = events

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

    def __del__(self) -> None:
        self.events.append('destroy_mcts')


class _LifecycleRoot:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __del__(self) -> None:
        self.events.append('release_root')


def test_self_play_constructs_selected_inference_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParams
    fake_alpha_zero_cpp.MCTS = _FakeMcts
    fake_alpha_zero_cpp.MCTSParams = _FakeMctsParams
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    training_args = copy.deepcopy(TRAINING_ARGS)
    training_args.self_play.use_inference_cache = False
    training_args.self_play.inference_cache_capacity = 0
    self_play = SelfPlayCpp(device_id=0, args=training_args)

    self_play._set_mcts(iteration=50)

    assert _FakeMcts.use_inference_cache is False


def test_iteration_update_releases_roots_and_reuses_mcts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    self_play = object.__new__(SelfPlayCpp)
    self_play.dataset = []
    self_play.iteration = 0
    self_play.self_play_games = [SimpleNamespace(already_expanded_node=_LifecycleRoot(events))]
    self_play.mcts = _LifecycleMcts(events)
    previous_mcts = self_play.mcts

    def update_mcts(iteration: int) -> None:
        assert iteration == 1
        assert self_play.mcts is previous_mcts
        events.append('update_mcts')

    monkeypatch.setattr(self_play, '_set_mcts', update_mcts)

    self_play.update_iteration(1)

    assert events == ['release_root', 'update_mcts']
