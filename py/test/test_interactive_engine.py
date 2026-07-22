from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("AlphaZeroCpp")
from AlphaZeroCpp import InteractiveSearchParams

from src.eval.InteractiveEngine import AnalysisMode, InferenceTarget, InteractiveEngine
from src.settings import CurrentGame


class _UniformModel(nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = inputs.size(0)
        policy = torch.full(
            (batch_size, self.action_size),
            1.0 / self.action_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        outcome = torch.tensor(
            [0.6, 0.3, 0.1], dtype=inputs.dtype, device=inputs.device
        )
        return policy, outcome.repeat(batch_size, 1)


class _InvalidOutcomeModel(_UniformModel):
    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        policy, _ = super().forward(inputs)
        outcome = torch.tensor(
            [1.2, -0.1, -0.1], dtype=inputs.dtype, device=inputs.device
        )
        return policy, outcome.repeat(inputs.size(0), 1)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    channels, rows, columns = CurrentGame.representation_shape
    model = _UniformModel(CurrentGame.action_size).eval()
    traced = torch.jit.trace(model, torch.zeros((1, channels, rows, columns)))
    path = tmp_path / "interactive.jit.pt"
    traced.save(str(path))
    return path


@pytest.fixture
def engine(model_path: Path) -> InteractiveEngine:
    return InteractiveEngine(
        model_path=str(model_path),
        device_id=0,
        parallel_searches=2,
        c_param=1.0,
        maximum_batch_size=2,
        outstanding_batches_per_worker=2,
        batch_collection_timeout_microseconds=50,
        cache_capacity=10_000,
        inference_target=InferenceTarget.CPU,
    )


@pytest.fixture
def invalid_engine(tmp_path: Path) -> InteractiveEngine:
    channels, rows, columns = CurrentGame.representation_shape
    model = _InvalidOutcomeModel(CurrentGame.action_size).eval()
    traced = torch.jit.trace(model, torch.zeros((1, channels, rows, columns)))
    path = tmp_path / "invalid-outcome.jit.pt"
    traced.save(str(path))
    return InteractiveEngine(
        model_path=str(path),
        device_id=0,
        parallel_searches=8,
        c_param=1.0,
        inference_workers=2,
        outstanding_batches_per_worker=2,
        inference_target=InferenceTarget.CPU,
    )


def test_policy_preserves_wdl_and_bypasses_search(engine: InteractiveEngine) -> None:
    game = engine.new_game(chess.STARTING_FEN, ())
    result = game.analyze(AnalysisMode.POLICY)

    assert result.searches == 0
    assert result.maximum_depth == 0
    assert result.outcome is not None
    assert result.outcome.win == pytest.approx(0.6)
    assert result.outcome.draw == pytest.approx(0.3)
    assert result.outcome.loss == pytest.approx(0.1)
    assert result.chosen_move_uci == min(
        candidate.move_uci for candidate in result.candidates
    )


def test_outstanding_batch_limit_is_validated(model_path: Path) -> None:
    with pytest.raises(ValueError, match="outstanding_batches_per_worker"):
        InteractiveEngine(
            model_path=str(model_path),
            device_id=0,
            parallel_searches=2,
            c_param=1.0,
            outstanding_batches_per_worker=3,
            inference_target=InferenceTarget.CPU,
        )


def test_native_search_params_default_to_pipelined_inference() -> None:
    search_parameters = InteractiveSearchParams(1.0, 2, 64)

    assert search_parameters.outstanding_batches_per_worker == 2


def test_fixed_search_reuses_selected_subtree_and_recovers_from_history(
    engine: InteractiveEngine,
) -> None:
    starting_fen = chess.STARTING_FEN
    game = engine.new_game(starting_fen, ())
    result = game.analyze(AnalysisMode.MCTS, search_limit=32)

    assert result.searches == 32
    assert result.candidates == tuple(
        sorted(
            result.candidates,
            key=lambda candidate: (
                -candidate.visits,
                -candidate.policy_prior,
                candidate.move_uci,
            ),
        )
    )
    assert result.chosen_move_uci == result.candidates[0].move_uci
    assert sum(
        candidate.visit_share for candidate in result.candidates
    ) == pytest.approx(1.0)
    assert all(
        candidate.mean_value is None or -1.0 <= candidate.mean_value <= 1.0
        for candidate in result.candidates
    )

    game.apply_move(result.chosen_move_uci)
    retained_visits = game.root_visits
    assert retained_visits == result.candidates[0].visits

    recovered = engine.new_game(starting_fen, (result.chosen_move_uci,))
    assert recovered.fen == game.fen
    assert recovered.root_visits == 0


def test_human_reply_reuses_a_less_searched_descendant(
    engine: InteractiveEngine,
) -> None:
    game = engine.new_game(chess.STARTING_FEN, ())
    engine_result = game.analyze(AnalysisMode.MCTS, search_limit=96)
    game.apply_move(engine_result.chosen_move_uci)

    reply_analysis = game.analyze(AnalysisMode.MCTS, search_limit=64)
    searched_replies = tuple(
        candidate for candidate in reply_analysis.candidates if candidate.visits > 0
    )
    assert len(searched_replies) > 1
    human_reply = min(
        searched_replies,
        key=lambda candidate: (candidate.visits, candidate.move_uci),
    )

    game.apply_move(human_reply.move_uci)
    assert game.root_visits == human_reply.visits

    additional_searches = 16
    game.analyze(AnalysisMode.MCTS, search_limit=additional_searches)
    assert game.root_visits == human_reply.visits + additional_searches


def test_history_replay_and_apply_move_reject_illegal_continuations(
    engine: InteractiveEngine,
) -> None:
    starting_fen = chess.STARTING_FEN
    repetition = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")
    terminal_game = engine.new_game(starting_fen, repetition)

    with pytest.raises(ValueError, match="Illegal UCI move"):
        engine.new_game(starting_fen, ("e2e5",))
    with pytest.raises(ValueError, match="game over"):
        terminal_game.apply_move("e2e4")


def test_timed_search_drains_direct_workers(engine: InteractiveEngine) -> None:
    game = engine.new_game(chess.STARTING_FEN, ())
    result = game.analyze(AnalysisMode.MCTS, time_limit_seconds=1)
    metrics = engine.inference_metrics()

    assert result.searches > 0
    assert 900 <= result.elapsed_milliseconds <= 1_100
    assert game.root_visits == result.searches
    assert 0 < metrics.model_positions <= result.searches
    assert 0.0 < metrics.direct_worker_utilization <= 1.0
    assert metrics.tree_selection_nanoseconds > 0
    assert metrics.board_encoding_nanoseconds > 0


def test_inference_failure_cancels_every_tree_reservation(
    invalid_engine: InteractiveEngine,
) -> None:
    game = invalid_engine.new_game(chess.STARTING_FEN, ())

    with pytest.raises(RuntimeError, match="WDL output"):
        game.analyze(AnalysisMode.MCTS, search_limit=16)
    assert game.root_visits == 0

    with pytest.raises(RuntimeError, match="WDL output"):
        game.analyze(AnalysisMode.MCTS, search_limit=16)
    assert game.root_visits == 0
