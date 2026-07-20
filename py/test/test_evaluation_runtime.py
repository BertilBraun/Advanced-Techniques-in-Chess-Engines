from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import pytest

import src.eval.ModelEvaluationCpp as evaluation_module
from src.eval.ModelEvaluationCpp import ModelEvaluation
from src.eval.ModelEvaluationPy import EvaluationModel, Results
from src.experiment.evaluation_protocol import GameRecord, ScheduledGame
from src.train.TrainingArgs import TrainingArgs


@dataclass(frozen=True)
class _EvaluationSettings:
    inference_cache_capacity: int
    mcts_threads: int = 1
    maximum_game_plies: int = 200
    stockfish_binary_path: str | None = None
    stockfish_hash_mib: int = 128


@dataclass(frozen=True)
class _EvaluationTrainingArguments:
    save_path: str
    evaluation: _EvaluationSettings


@dataclass(frozen=True)
class _FakeInferenceClientParameters:
    device_id: int
    model_path: str
    maximum_batch_size: int
    timeout_microseconds: int
    cache_capacity: int


@dataclass(frozen=True)
class _FakeMctsBoard:
    root: object
    should_run_full_search: bool


def test_model_comparison_uses_configured_cache_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    iteration = 4
    current_model_path = tmp_path / f'model_{iteration}.jit.pt'
    opponent_model_path = tmp_path / 'model_0.jit.pt'
    current_model_path.touch()
    opponent_model_path.touch()

    cache_capacities: list[int] = []

    class FakeMcts:
        def __init__(
            self,
            inference_parameters: _FakeInferenceClientParameters,
            mcts_parameters: str,
        ) -> None:
            assert mcts_parameters == 'mcts-parameters'
            cache_capacities.append(inference_parameters.cache_capacity)

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

    def mcts_parameters(_: ModelEvaluation) -> str:
        return 'mcts-parameters'

    fake_alpha_zero_cpp = ModuleType('AlphaZeroCpp')
    fake_alpha_zero_cpp.InferenceClientParams = _FakeInferenceClientParameters
    fake_alpha_zero_cpp.MCTS = FakeMcts
    fake_alpha_zero_cpp.MCTSBoard = _FakeMctsBoard
    monkeypatch.setitem(sys.modules, 'AlphaZeroCpp', fake_alpha_zero_cpp)
    monkeypatch.setattr(ModelEvaluation, 'mcts_args', property(mcts_parameters))
    monkeypatch.setattr(evaluation_module, '_play_paired_models_search', fake_paired_match)

    model_evaluation = ModelEvaluation.__new__(ModelEvaluation)
    model_evaluation.iteration = iteration
    model_evaluation.device_id = 0
    model_evaluation.args = cast(
        TrainingArgs,
        _EvaluationTrainingArguments(
            save_path=str(tmp_path),
            evaluation=_EvaluationSettings(inference_cache_capacity=50_000),
        ),
    )
    model_evaluation.paired_schedule = ()

    model_evaluation.play_two_models_paired(tmp_path / 'model_0.pt')

    assert cache_capacities == [50_000, 50_000]


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
                inference_cache_capacity=50_000,
                stockfish_binary_path='/stockfish',
                stockfish_hash_mib=64,
            ),
        ),
    )
    monkeypatch.setattr(model_evaluation, 'play_vs_evaluation_model', finish_without_games)

    model_evaluation.play_vs_stockfish(level=2)

    assert stockfish.configurations == [
        ('Skill Level', 2),
        ('Threads', 1),
        ('Hash', 64),
    ]
    assert stockfish.was_quit
