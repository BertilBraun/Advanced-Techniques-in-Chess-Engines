from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
from AlphaZeroCpp import (
    EvalMCTS,
    EvalMCTSNode,
    EvalMCTSParams,
    InferenceClientParams,
    new_eval_root_with_history,
)

from src.games.chess.repetition_history import (
    REPETITION_HISTORY_PLIES,
    RepetitionHistory,
    bounded_repetition_history,
)
from web_play.contracts import (
    AnalysisLimit,
    AnalysisMode,
    AnalysisResult,
    CandidateMove,
    CountedAnalysis,
    SearchMetrics,
    TimedAnalysis,
)

_SEARCH_CHUNK_SIZE = 8


@dataclass(frozen=True)
class NativeEngineConfiguration:
    model_path: str
    device_id: int
    search_threads: int
    cache_capacity: int
    maximum_batch_size: int
    inference_batch_timeout_microseconds: int
    exploration_constant: float

    @classmethod
    def for_model(
        cls, model_path: str, search_threads: int, device_id: int = 0
    ) -> NativeEngineConfiguration:
        if search_threads < 1:
            raise ValueError("search_threads must be positive.")
        return cls(
            model_path=model_path,
            device_id=device_id,
            search_threads=search_threads,
            cache_capacity=250_000,
            maximum_batch_size=256,
            inference_batch_timeout_microseconds=500,
            exploration_constant=1.0,
        )


class NativeInteractiveEngine:
    """Long-lived owner of the model, inference cache, and native worker pools."""

    def __init__(self, configuration: NativeEngineConfiguration) -> None:
        inference_parameters = InferenceClientParams(
            device_id=configuration.device_id,
            currentModelPath=configuration.model_path,
            maxBatchSize=configuration.maximum_batch_size,
            microsecondsTimeoutInferenceThread=configuration.inference_batch_timeout_microseconds,
            cacheCapacity=configuration.cache_capacity,
        )
        search_parameters = EvalMCTSParams(
            c_param=configuration.exploration_constant,
            num_threads=configuration.search_threads,
        )
        self._search = EvalMCTS(inference_parameters, search_parameters)

    def new_game(
        self, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> NativeInteractiveGame:
        return NativeInteractiveGame(self._search, starting_fen, moves_uci)


class NativeInteractiveGame:
    """Serially-used game with a bounded native history and reusable MCTS root."""

    def __init__(
        self, search: EvalMCTS, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> None:
        self._search = search
        self._board = _replay_history(starting_fen, moves_uci)
        self._history = bounded_repetition_history(
            self._board, REPETITION_HISTORY_PLIES
        )
        self._root = _new_root(self._history)

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def is_game_over(self) -> bool:
        return self._board.is_game_over(claim_draw=True)

    @property
    def result(self) -> str | None:
        return self._board.result(claim_draw=True) if self.is_game_over else None

    def apply_move(self, move_uci: str) -> None:
        try:
            move = self._board.parse_uci(move_uci)
        except ValueError as error:
            raise ValueError(
                f"Illegal UCI move {move_uci!r} for {self._board.fen()}."
            ) from error

        matching_child_index = next(
            (
                index
                for index, child in enumerate(self._root.children)
                if child.move == move_uci
            ),
            None,
        )
        self._board.push(move)
        self._history = bounded_repetition_history(
            self._board, REPETITION_HISTORY_PLIES
        )
        if matching_child_index is None:
            self._root = _new_root(self._history)
        else:
            self._root = self._root.make_new_root(matching_child_index)

    def analyze(self, limit: AnalysisLimit) -> AnalysisResult:
        if self.is_game_over:
            raise ValueError("Cannot analyze a finished game.")
        match limit:
            case TimedAnalysis(mode=mode, time_limit_seconds=seconds):
                return self._analyze_timed(mode, seconds)
            case CountedAnalysis(mode=mode, searches=searches):
                return self._analyze_counted(mode, searches)

    def _analyze_timed(self, mode: AnalysisMode, seconds: int) -> AnalysisResult:
        if mode is AnalysisMode.POLICY:
            return self._analyze_policy()

        started = time.monotonic()
        deadline = started + seconds
        visits_before = self._root.visits
        while self._root.visits == visits_before or time.monotonic() < deadline:
            remaining_seconds = deadline - time.monotonic()
            chunk_size = 1 if remaining_seconds < 0.05 else _SEARCH_CHUNK_SIZE
            self._search.eval_search(self._root, chunk_size)
        return self._build_result(
            mode=mode,
            searches=self._root.visits - visits_before,
            started=started,
            root=self._root,
        )

    def _analyze_counted(self, mode: AnalysisMode, searches: int) -> AnalysisResult:
        if mode is AnalysisMode.POLICY:
            return self._analyze_policy()
        started = time.monotonic()
        visits_before = self._root.visits
        self._search.eval_search(self._root, searches)
        return self._build_result(
            mode=mode,
            searches=self._root.visits - visits_before,
            started=started,
            root=self._root,
        )

    def _analyze_policy(self) -> AnalysisResult:
        started = time.monotonic()
        policy_root = _new_root(self._history)
        self._search.eval_search(policy_root, 1)
        return self._build_result(
            mode=AnalysisMode.POLICY,
            searches=1,
            started=started,
            root=policy_root,
        )

    def _build_result(
        self,
        mode: AnalysisMode,
        searches: int,
        started: float,
        root: EvalMCTSNode,
    ) -> AnalysisResult:
        candidates = _candidate_moves(root, mode)
        if not candidates:
            raise ValueError("The native engine returned no legal candidate moves.")
        root_value = root.result_sum / root.visits if root.visits > 0 else 0.0
        return AnalysisResult(
            chosen_move_uci=candidates[0].move_uci,
            root_value=_clamp_value(root_value),
            outcome_prediction=None,
            candidates=candidates,
            metrics=SearchMetrics(
                searches=searches,
                maximum_depth=root.max_depth,
                elapsed_milliseconds=round((time.monotonic() - started) * 1000),
            ),
            principal_variation=_principal_variation(root)
            if mode is AnalysisMode.MCTS
            else None,
        )


def _replay_history(starting_fen: str, moves_uci: tuple[str, ...]) -> chess.Board:
    try:
        board = chess.Board(starting_fen)
    except ValueError as error:
        raise ValueError(f"Invalid starting FEN: {starting_fen!r}.") from error
    for ply, move_uci in enumerate(moves_uci, start=1):
        try:
            move = board.parse_uci(move_uci)
        except ValueError as error:
            raise ValueError(f"Illegal UCI move {move_uci!r} at ply {ply}.") from error
        board.push(move)
    return board


def _new_root(history: RepetitionHistory) -> EvalMCTSNode:
    return new_eval_root_with_history(history.starting_fen, history.moves_uci)


def _candidate_moves(
    root: EvalMCTSNode, mode: AnalysisMode
) -> tuple[CandidateMove, ...]:
    total_child_visits = sum(child.visits for child in root.children)
    candidates = tuple(
        CandidateMove(
            move_uci=child.move,
            policy_prior=child.policy,
            visits=child.visits,
            visit_share=child.visits / total_child_visits
            if total_child_visits > 0
            else 0.0,
            mean_search_value=(
                _clamp_value(-child.result_sum / child.visits)
                if child.visits > 0
                else None
            ),
        )
        for child in root.children
    )
    if mode is AnalysisMode.POLICY:
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (-candidate.policy_prior, candidate.move_uci),
            )
        )
    return tuple(
        sorted(
            candidates, key=lambda candidate: (-candidate.visits, candidate.move_uci)
        )
    )


def _principal_variation(root: EvalMCTSNode) -> tuple[str, ...] | None:
    moves: list[str] = []
    node = root
    while node.children:
        visited_children = tuple(child for child in node.children if child.visits > 0)
        if not visited_children:
            break
        child = min(
            visited_children, key=lambda candidate: (-candidate.visits, candidate.move)
        )
        moves.append(child.move)
        node = child
    return tuple(moves) or None


def _clamp_value(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("The native engine returned a non-finite value.")
    return min(1.0, max(-1.0, value))
