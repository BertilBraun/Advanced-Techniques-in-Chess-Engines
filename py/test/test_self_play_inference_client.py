import copy
from pathlib import Path
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace

import pytest

from src.self_play.SelfPlay import SelfPlay, SelfPlayGame, has_positive_visit_counts
from src.self_play.chess_completed_game import ChessCompletedGamePublisher
from src.settings import TRAINING_ARGS
from src.train.TrainingArgs import DirectInferenceParams


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
class _FakeBatchedInferenceParameters:
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int


class _FakeMcts:
    inference_parameters: _FakeBatchedInferenceParameters | None = None

    def __init__(
        self,
        client_args: _FakeInferenceClientParams,
        mcts_args: _FakeMctsParams,
        inference_parameters: _FakeBatchedInferenceParameters | None,
        initial_model_version: int,
    ) -> None:
        assert client_args.maxBatchSize == 64
        assert mcts_args.num_full_searches > mcts_args.num_parallel_searches
        _FakeMcts.inference_parameters = inference_parameters
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

    def reset(self) -> None:
        self.events.append('reset_root')

    def __del__(self) -> None:
        self.events.append('release_root')


def test_self_play_constructs_direct_inference_client_during_search_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParams
    fake_alpha_zero_cpp.BatchedInferenceParameters = _FakeBatchedInferenceParameters
    fake_alpha_zero_cpp.MCTS = _FakeMcts
    fake_alpha_zero_cpp.MCTSParams = _FakeMctsParams
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    search = TRAINING_ARGS.self_play.search.validated_copy(update={'num_searches_per_turn': 600})
    self_play_args = TRAINING_ARGS.self_play.validated_copy(
        update={
            'initial_num_searches_per_turn': 100,
            'search': search.model_dump(mode='json'),
            'search_warmup_model_versions': 100,
        }
    )
    training_args = TRAINING_ARGS.validated_copy(
        update={'self_play': self_play_args.model_dump(mode='json'), 'save_path': str(tmp_path)}
    )
    self_play = SelfPlay(
        device_id=0,
        args=training_args,
        completed_game_publisher=ChessCompletedGamePublisher(tmp_path, 0, 0),
    )

    self_play._set_mcts(iteration=50)

    assert _FakeMcts.inference_parameters == _FakeBatchedInferenceParameters(2, 64, 2)
    assert self_play.search_schedule(0).num_full_searches == 100
    assert self_play.search_schedule(1).num_full_searches == 105
    assert self_play.search_schedule(100).num_full_searches == 600


def test_self_play_constructs_direct_inference_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParams
    fake_alpha_zero_cpp.BatchedInferenceParameters = _FakeBatchedInferenceParameters
    fake_alpha_zero_cpp.MCTS = _FakeMcts
    fake_alpha_zero_cpp.MCTSParams = _FakeMctsParams
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)

    direct_inference = DirectInferenceParams(
        inference_workers=2,
        inference_batch_size=64,
        outstanding_batches_per_worker=1,
    )
    self_play_args = TRAINING_ARGS.self_play.validated_copy(
        update={'inference': direct_inference.model_dump(mode='json')}
    )
    training_args = TRAINING_ARGS.validated_copy(
        update={'self_play': self_play_args.model_dump(mode='json'), 'save_path': str(tmp_path)}
    )
    self_play = SelfPlay(
        device_id=0,
        args=training_args,
        completed_game_publisher=ChessCompletedGamePublisher(tmp_path, 0, 0),
    )

    self_play._set_mcts(iteration=50)

    assert _FakeMcts.inference_parameters == _FakeBatchedInferenceParameters(2, 64, 1)


def test_model_refresh_retains_game_state_and_resets_search_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    self_play = object.__new__(SelfPlay)
    self_play.args = copy.deepcopy(TRAINING_ARGS.self_play)
    self_play.search_warmup_iterations = TRAINING_ARGS.self_play.search_warmup_model_versions
    self_play.endgame_shortcut_fade_iterations = TRAINING_ARGS.self_play.endgame_shortcut_fade_model_versions
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
    monkeypatch.setattr('src.self_play.SelfPlay.log_scalar', lambda *_args: None)

    self_play.refresh_model(1, 'updated.jit.pt')

    assert self_play.mcts is previous_mcts
    assert game.already_expanded_node is root
    assert len(self_play.dataset) == 1
    assert self_play.completed_searches == 19
    assert self_play.search_schedule_state == self_play.search_schedule(0)
    assert self_play.model_version == 1
    assert self_play.model_refresh_acknowledgements == [0, 1]
    assert events == ['refresh:1:updated.jit.pt', 'reset_root']


def test_failed_model_refresh_is_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    self_play = object.__new__(SelfPlay)
    self_play.args = copy.deepcopy(TRAINING_ARGS.self_play)
    self_play.search_warmup_iterations = TRAINING_ARGS.self_play.search_warmup_model_versions
    self_play.endgame_shortcut_fade_iterations = TRAINING_ARGS.self_play.endgame_shortcut_fade_model_versions
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
