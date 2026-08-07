from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from torch import multiprocessing as mp

import src.games.chess.evaluation.process as evaluation_process_module
import src.games.chess.evaluation.model as evaluation_module
from src.games.chess.evaluation.process import (
    _activate_evaluation_device,
    _model_engine_condition,
    _reap_evaluation_tasks,
    _terminate_evaluation_tasks,
)
from src.games.chess.evaluation.model import ModelEvaluation
from src.games.chess.evaluation.types import EvaluationModel, EvaluationMove, PairedEvaluationModel, Results
from src.experiment.evaluation_protocol import GameRecord, ScheduledGame
from src.games.chess.visit_policy import action_probabilities
from src.games.chess.board import ChessBoard
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.training.configuration import (
    BatchedInferenceParams,
    TrainingArgs,
)


@dataclass(frozen=True)
class _EvaluationSettings:
    inference: BatchedInferenceParams
    parallel_searches: int = 1
    maximum_game_plies: int | None = 200
    stockfish_binary_path: str | None = None
    stockfish_hash_mib: int = 128


@pytest.mark.parametrize('visit_counts', ([], [(1, 0), (2, 0)]))
def test_action_probabilities_reject_zero_visit_mass(visit_counts: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match='at least one positive visit'):
        action_probabilities(visit_counts, CHESS_STATE_CONTRACT.action_size)


@dataclass(frozen=True)
class _EvaluationTrainingArguments:
    save_path: str
    evaluation: _EvaluationSettings
    network: str = 'unused'


@dataclass(frozen=True)
class _FakeInferenceConfiguration:
    device_id: int
    model_path: str


@dataclass(frozen=True)
class _FakeChessSelfPlaySearchRequest:
    root: object
    full_search: bool


@dataclass(frozen=True)
class _FakeBatchedInferenceParameters:
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int


class _FakeEvaluationProcess:
    def __init__(self, process_id: int, alive: bool, exit_code: int | None) -> None:
        self.pid = process_id
        self.exitcode = exit_code
        self.alive = alive
        self.join_count = 0
        self.was_terminated = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_count += 1

    def terminate(self) -> None:
        self.was_terminated = True
        self.alive = False
        self.exitcode = -15


def test_evaluation_task_activates_assigned_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    activated_devices: list[int] = []
    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.device_id = 2
    monkeypatch.setattr(evaluation_process_module, 'USE_GPU', True)
    monkeypatch.setattr(evaluation_process_module.torch.cuda, 'set_device', activated_devices.append)

    _activate_evaluation_device(model_evaluation)

    assert activated_devices == [2]


def test_failed_evaluation_task_is_detected_while_another_task_is_alive() -> None:
    active_process = _FakeEvaluationProcess(process_id=10, alive=True, exit_code=None)
    failed_process = _FakeEvaluationProcess(process_id=11, alive=False, exit_code=1)
    processes = cast(list[mp.Process], [active_process, failed_process])

    with pytest.raises(RuntimeError, match='11: 1'):
        _reap_evaluation_tasks(processes)

    assert processes == [active_process]
    assert failed_process.join_count == 1


def test_evaluation_task_cleanup_terminates_active_peers() -> None:
    active_process = _FakeEvaluationProcess(process_id=10, alive=True, exit_code=None)
    completed_process = _FakeEvaluationProcess(process_id=11, alive=False, exit_code=0)
    processes = cast(list[mp.Process], [active_process, completed_process])

    _terminate_evaluation_tasks(processes)

    assert active_process.was_terminated
    assert active_process.join_count == 1
    assert not completed_process.was_terminated
    assert completed_process.join_count == 1
    assert processes == []


def test_model_comparison_uses_configured_inference_client_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    iteration = 4
    current_model_path = tmp_path / f'model_{iteration}.jit.pt'
    opponent_model_path = tmp_path / 'model_0.jit.pt'
    current_model_path.touch()
    opponent_model_path.touch()

    inference_clients: list[_FakeBatchedInferenceParameters | None] = []

    class FakeChessSelfPlaySearch:
        def __init__(
            self,
            runtime_parameters: _FakeInferenceConfiguration,
            search_parameters: str,
            inference_parameters: _FakeBatchedInferenceParameters | None,
        ) -> None:
            assert search_parameters == 'search-parameters'
            assert runtime_parameters.model_path.endswith('.jit.pt')
            inference_clients.append(inference_parameters)

    def fake_paired_match(
        *,
        iteration: int,
        candidate_model: EvaluationModel,
        opponent_model: EvaluationModel,
        schedule: tuple[ScheduledGame, ...],
        maximum_game_plies: int,
        name: str,
    ) -> tuple[Results, tuple[GameRecord, ...]]:
        assert iteration == 4
        assert callable(candidate_model)
        assert callable(opponent_model)
        assert schedule == ()
        assert maximum_game_plies == 200
        assert name == opponent_model_path.name
        return Results(0, 0, 0), ()

    def search_parameters(_: ModelEvaluation) -> str:
        return 'search-parameters'

    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceConfiguration = _FakeInferenceConfiguration
    fake_alpha_zero_cpp.BatchedInferenceParameters = _FakeBatchedInferenceParameters
    fake_alpha_zero_cpp.ChessSelfPlaySearch = FakeChessSelfPlaySearch
    fake_alpha_zero_cpp.ChessSelfPlaySearchRequest = _FakeChessSelfPlaySearchRequest
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)
    monkeypatch.setattr(ModelEvaluation, 'self_play_search_parameters', property(search_parameters))
    monkeypatch.setattr(evaluation_module, 'play_paired_models', fake_paired_match)

    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.iteration = iteration
    model_evaluation.device_id = 0
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path=str(tmp_path),
            evaluation=_EvaluationSettings(
                inference=BatchedInferenceParams(
                    inference_workers=1,
                    inference_batch_size=64,
                    outstanding_batches_per_worker=1,
                ),
            ),
        ),
    )
    model_evaluation.evaluation_args = model_evaluation.args.evaluation
    model_evaluation.paired_schedule = ()

    model_evaluation.play_two_models_paired(tmp_path / 'model_0.pt')

    expected_direct_parameters = _FakeBatchedInferenceParameters(1, 64, 1)
    assert inference_clients == [
        expected_direct_parameters,
        expected_direct_parameters,
    ]


def test_direct_evaluation_provenance_records_scheduler_topology() -> None:
    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.num_searches_per_turn = 64
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path='unused',
            evaluation=_EvaluationSettings(
                inference=BatchedInferenceParams(
                    inference_workers=1,
                    inference_batch_size=64,
                    outstanding_batches_per_worker=1,
                ),
                parallel_searches=1,
            ),
        ),
    )
    model_evaluation.evaluation_args = model_evaluation.args.evaluation

    condition = _model_engine_condition(model_evaluation, 'a' * 64, 'candidate-model-1')

    assert condition.search_limit_name == 'mcts_root_visits'
    assert condition.search_limit_value == 64
    assert condition.threads == 1
    assert {(setting.name, setting.value) for setting in condition.settings} == {
        ('InferenceScheduler', 'direct_multi_tree'),
        ('InferenceWorkers', '1'),
        ('InferenceBatchSize', '64'),
        ('OutstandingBatchesPerWorker', '1'),
        ('ParallelSearchesPerTree', '1'),
    }


def test_policy_evaluation_uses_native_batched_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_paths: list[Path] = []

    class FakeNativeSearch:
        def inference_with_history(
            self,
            histories: list[tuple[str, tuple[str, ...]]],
        ) -> list[tuple[list[tuple[int, float]], float]]:
            raise AssertionError(f'No games should be evaluated in this unit test: {histories}')

    def create_native_search(_: ModelEvaluation, inference_path: Path) -> FakeNativeSearch:
        inference_paths.append(inference_path)
        return FakeNativeSearch()

    def finish_without_games(
        iteration: int,
        candidate_model: PairedEvaluationModel,
        opponent_model: PairedEvaluationModel,
        schedule: tuple[ScheduledGame, ...],
        maximum_game_plies: int | None,
        name: str,
    ) -> tuple[Results, tuple[GameRecord, ...]]:
        assert iteration == 7
        assert callable(candidate_model)
        assert callable(opponent_model)
        assert schedule == ()
        assert maximum_game_plies == 200
        assert name == 'policy_vs_random_paired'
        return Results(0, 0, 0), ()

    monkeypatch.setattr(ModelEvaluation, '_create_search', create_native_search)
    monkeypatch.setattr(evaluation_module, 'play_paired_models', finish_without_games)

    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.iteration = 7
    model_evaluation.num_games = 0
    model_evaluation.paired_schedule = ()
    model_evaluation.device_id = 2
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path='training-output',
            evaluation=_EvaluationSettings(
                inference=BatchedInferenceParams(
                    inference_workers=1,
                    inference_batch_size=64,
                    outstanding_batches_per_worker=1,
                ),
            ),
            network='network-settings',
        ),
    )
    model_evaluation.evaluation_args = model_evaluation.args.evaluation

    assert model_evaluation.play_policy_vs_random() == Results(0, 0, 0)
    assert inference_paths == [Path('training-output/model_7.jit.pt')]


class _FakeStockfish:
    def __init__(self) -> None:
        self.configurations: list[tuple[str, int]] = []
        self.was_quit = False

    def configure(self, options: Mapping[str, int]) -> None:
        self.configurations.extend(options.items())

    def quit(self) -> None:
        self.was_quit = True


def test_skill_level_stockfish_uses_configured_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stockfish = _FakeStockfish()

    def open_stockfish(_: str) -> _FakeStockfish:
        return stockfish

    def finish_without_games(_: EvaluationModel, name: str) -> Results:
        assert name == 'stockfish_level_2'
        return Results(0, 0, 0)

    monkeypatch.setattr(evaluation_module.chess.engine.SimpleEngine, 'popen_uci', open_stockfish)

    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path='unused',
            evaluation=_EvaluationSettings(
                inference=BatchedInferenceParams(
                    inference_workers=1,
                    inference_batch_size=64,
                    outstanding_batches_per_worker=1,
                ),
                stockfish_binary_path='/stockfish',
                stockfish_hash_mib=64,
            ),
        ),
    )
    model_evaluation.evaluation_args = model_evaluation.args.evaluation
    monkeypatch.setattr(model_evaluation, 'play_vs_evaluation_model', finish_without_games)

    model_evaluation.play_vs_stockfish(level=2)

    assert stockfish.configurations == [
        ('Skill Level', 2),
        ('Threads', 1),
        ('Hash', 64),
    ]
    assert stockfish.was_quit


def test_skill_level_stockfish_restarts_after_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecoveringStockfish(_FakeStockfish):
        def __init__(self, fail: bool) -> None:
            super().__init__()
            self.fail = fail
            self.was_closed = False

        def play(
            self,
            board: object,
            limit: object,
            *,
            ponder: bool,
        ) -> SimpleNamespace:
            assert board is current_board.board
            assert limit is not None
            assert not ponder
            if self.fail:
                raise evaluation_module.chess.engine.EngineError('poisoned session')
            return SimpleNamespace(move=current_board.get_valid_moves()[0])

        def close(self) -> None:
            self.was_closed = True

    first_engine = RecoveringStockfish(fail=True)
    second_engine = RecoveringStockfish(fail=False)
    engines = iter((first_engine, second_engine))
    current_board = ChessBoard()

    monkeypatch.setattr(
        evaluation_module.chess.engine.SimpleEngine,
        'popen_uci',
        lambda _: next(engines),
    )

    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path='unused',
            evaluation=_EvaluationSettings(
                inference=BatchedInferenceParams(
                    inference_workers=1,
                    inference_batch_size=64,
                    outstanding_batches_per_worker=1,
                ),
                stockfish_binary_path='/stockfish',
                stockfish_hash_mib=64,
            ),
        ),
    )
    model_evaluation.evaluation_args = model_evaluation.args.evaluation

    def evaluate_once(evaluator: PairedEvaluationModel, _: str) -> Results:
        decisions = evaluator([current_board])
        assert len(decisions) == 1
        assert isinstance(decisions[0], EvaluationMove)
        return Results(1, 0, 0)

    monkeypatch.setattr(model_evaluation, 'play_vs_evaluation_model', evaluate_once)

    assert model_evaluation.play_vs_stockfish(level=3) == Results(1, 0, 0)
    assert first_engine.was_closed
    assert second_engine.was_quit
