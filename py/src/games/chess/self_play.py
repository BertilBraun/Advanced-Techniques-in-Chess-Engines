from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from src.games.chess.board import ChessBoard
from src.games.chess.configuration import ChessSelfPlayConfiguration
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.repetition_history import REPETITION_HISTORY_PLIES, bounded_repetition_history
from src.games.contracts import WdlTarget
from src.self_play.active_game import CompletedGame, ContinuingGame
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
)
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.worker import GameSelfPlayPolicy
from src.util.tensorboard import log_scalar


if TYPE_CHECKING:
    from AlphaZeroCpp import (
        ChessSearchRoot,
        ChessSelfPlaySearch,
        ChessSelfPlaySearchRequest,
        ChessSelfPlaySearchResult,
        InferenceStatistics,
        SelfPlaySearchParameters,
        TimeInfo,
    )


@dataclass
class ChessSelfPlayGame:
    identity: GameIdentity
    board: ChessBoard
    started_at_seconds: float
    action_ids: list[int] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)
    root: ChessSearchRoot | None = None


SelfPlayGame = ChessSelfPlayGame


@dataclass(frozen=True)
class SelfPlayStatisticsSnapshot:
    model_version: int
    inference: InferenceStatistics
    timing: TimeInfo
    completed_searches: int


class ChessSelfPlayPolicy(
    GameSelfPlayPolicy[
        ChessSelfPlayGame,
        'ChessSelfPlaySearchRequest',
        'ChessSelfPlaySearchResult',
        SelfPlayStatisticsSnapshot | None,
    ]
):
    def __init__(
        self,
        device_id: int,
        configuration: ChessSelfPlayConfiguration,
        worker_id: int,
        random_seed: int,
    ) -> None:
        self.device_id = device_id
        self.configuration = configuration
        self.random = np.random.default_rng(random_seed + worker_id)
        self.model_generation: int | None = None
        self.resolved_parameters = self._resolve_parameters(0)
        self.search: ChessSelfPlaySearch | None = None
        self.completed_searches = 0

    def refresh_model(
        self,
        model_generation: int,
        model_path: Path,
        active_games: tuple[ChessSelfPlayGame, ...],
    ) -> None:
        if self.model_generation is not None and model_generation <= self.model_generation:
            raise ValueError('Chess model generation must increase on refresh.')
        self.resolved_parameters = self._resolve_parameters(model_generation)
        native_parameters = self._native_search_parameters()
        if self.search is None:
            from AlphaZeroCpp import (
                BatchedInferenceParameters,
                ChessSelfPlaySearch,
                InferenceConfiguration,
            )

            inference = self.configuration.inference
            self.search = ChessSelfPlaySearch(
                InferenceConfiguration(device_id=self.device_id, model_path=str(model_path)),
                native_parameters,
                inference_parameters=BatchedInferenceParameters(
                    inference.inference_workers,
                    inference.inference_batch_size,
                    inference.outstanding_batches_per_worker,
                ),
                initial_model_version=model_generation,
            )
        else:
            self.search.refresh_model(model_generation, str(model_path))
            self.search.update_search_schedule(native_parameters)
        self.model_generation = model_generation
        for game in active_games:
            if game.root is not None:
                game.root.reset()

    def snapshot_statistics(self, tensorboard_step: int) -> SelfPlayStatisticsSnapshot | None:
        if self.search is None or self.model_generation is None:
            return None
        inference, timing = self.search.inference_statistics()
        log_scalar(
            'inference/average_number_of_positions_in_inference_call',
            inference.averageNumberOfPositionsInInferenceCall,
            tensorboard_step,
        )
        log_scalar('timing/total_time_cpp', timing.totalTime, tensorboard_step)
        return SelfPlayStatisticsSnapshot(
            model_version=self.model_generation,
            inference=inference,
            timing=timing,
            completed_searches=self.completed_searches,
        )

    def new_game(self, identity: GameIdentity) -> ChessSelfPlayGame:
        if self.search is None:
            raise RuntimeError('A model must be loaded before creating a chess game.')
        while True:
            board = CHESS_STATE_CONTRACT.initial_position()
            action_ids: list[int] = []
            for _ in range(self.resolved_parameters.random_opening_plies):
                legal_actions = CHESS_STATE_CONTRACT.legal_action_ids(board)
                action_id = int(self.random.choice(legal_actions))
                action_ids.append(action_id)
                board = CHESS_STATE_CONTRACT.child_position(board, action_id)
                if CHESS_STATE_CONTRACT.terminal_wdl(board) is not None:
                    break
            if CHESS_STATE_CONTRACT.terminal_wdl(board) is None:
                return ChessSelfPlayGame(identity, board, time.time(), action_ids=action_ids)

    def build_search_request(self, game: ChessSelfPlayGame) -> ChessSelfPlaySearchRequest:
        from AlphaZeroCpp import ChessSelfPlaySearchRequest

        if self.search is None:
            raise RuntimeError('A model must be loaded before chess search starts.')
        if game.root is None:
            history = bounded_repetition_history(game.board.board, REPETITION_HISTORY_PLIES)
            game.root = self.search.new_root_with_history(history.starting_fen, history.moves_uci)
        full_search = self.random.random() < self.resolved_parameters.full_search_probability
        if full_search:
            game.root.discount(self.resolved_parameters.retained_root_visit_fraction)
        return ChessSelfPlaySearchRequest(game.root, full_search)

    def search_active_games(
        self,
        requests: tuple[ChessSelfPlaySearchRequest, ...],
    ) -> tuple[ChessSelfPlaySearchResult, ...]:
        if self.search is None:
            raise RuntimeError('A model must be loaded before chess search starts.')
        batch = self.search.search(list(requests), collect_statistics=False)
        self.completed_searches += batch.simulations_completed
        return tuple(batch.results)

    def advance_game(
        self,
        game: ChessSelfPlayGame,
        request: ChessSelfPlaySearchRequest,
        result: ChessSelfPlaySearchResult,
    ) -> ContinuingGame[ChessSelfPlayGame] | CompletedGame:
        if self.model_generation is None:
            raise RuntimeError('A model must be loaded before recording a chess move.')
        indexed_positive_visits = tuple(
            (native_index, SparseSearchVisit(action_id=action_id, visit_count=visit_count))
            for native_index, (action_id, visit_count) in enumerate(result.visits)
            if visit_count > 0
        )
        if not indexed_positive_visits:
            raise RuntimeError('Native chess search returned no visited action for a nonterminal root.')
        positive_visits = tuple(visit for _, visit in indexed_positive_visits)
        ply = len(game.action_ids)
        selected_visit_index = self._select_visit_index(positive_visits, ply)
        native_child_index, selected_visit = indexed_positive_visits[selected_visit_index]
        selected_action = selected_visit.action_id
        game.observations.append(
            SearchObservation(
                ply=ply,
                model_generation=self.model_generation,
                visits=positive_visits,
                root_value=result.root_value,
                selected_action_id=selected_action,
                full_search=request.full_search,
                sample_weight=self.resolved_parameters.primary_sample_weight,
                search_budget=(
                    self.resolved_parameters.full_searches
                    if request.full_search
                    else self.resolved_parameters.fast_searches
                ),
                minimum_root_visits=self.resolved_parameters.minimum_root_visits,
            )
        )
        game.action_ids.append(selected_action)
        game.board = CHESS_STATE_CONTRACT.child_position(game.board, selected_action)
        game.root = result.root.make_new_root(native_child_index)

        natural_wdl = CHESS_STATE_CONTRACT.terminal_wdl(game.board)
        if natural_wdl is not None:
            return CompletedGame(self._complete(game, natural_wdl, TerminationReason.NATURAL))
        maximum_plies = self.resolved_parameters.maximum_game_plies
        if maximum_plies is not None and len(game.action_ids) >= maximum_plies:
            wdl = CHESS_STATE_CONTRACT.adjudicated_wdl(game.board, TerminationReason.MAXIMUM_PLIES)
            return CompletedGame(self._complete(game, wdl, TerminationReason.MAXIMUM_PLIES))
        return ContinuingGame(game)

    def _select_visit_index(self, visits: tuple[SparseSearchVisit, ...], ply: int) -> int:
        counts = np.asarray([visit.visit_count for visit in visits], dtype=np.float64)
        if ply >= self.resolved_parameters.greedy_after_ply:
            return min(
                range(len(visits)),
                key=lambda index: (-visits[index].visit_count, visits[index].action_id),
            )
        progress = ply / self.resolved_parameters.greedy_after_ply
        temperature = (
            self.resolved_parameters.starting_temperature
            + (self.resolved_parameters.final_temperature - self.resolved_parameters.starting_temperature) * progress
        )
        probabilities = np.power(counts, 1.0 / temperature)
        probabilities /= probabilities.sum()
        return int(self.random.choice(len(visits), p=probabilities))

    @staticmethod
    def _complete(
        game: ChessSelfPlayGame,
        final_wdl: WdlTarget,
        reason: TerminationReason,
    ) -> CompletedSelfPlayGame:
        return CompletedSelfPlayGame(
            identity=game.identity,
            created_at_seconds=time.time(),
            generation_seconds=time.time() - game.started_at_seconds,
            action_ids=tuple(game.action_ids),
            observations=tuple(game.observations),
            final_wdl=final_wdl,
            termination_reason=reason,
        )

    def _resolve_parameters(self, model_generation: int) -> ResolvedSelfPlayParameters:
        maximum_game_plies = (
            None
            if self.configuration.maximum_game_plies is None
            else self.configuration.maximum_game_plies.value_at(model_generation)
        )
        return self.configuration.resolve(model_generation, maximum_game_plies)

    def _native_search_parameters(self) -> SelfPlaySearchParameters:
        from AlphaZeroCpp import SelfPlaySearchParameters

        parameters = self.resolved_parameters
        return SelfPlaySearchParameters(
            parallel_searches=parameters.parallel_searches,
            full_searches=parameters.full_searches,
            fast_searches=parameters.fast_searches,
            dirichlet_alpha=parameters.dirichlet_alpha,
            dirichlet_epsilon=parameters.dirichlet_epsilon,
            exploration_constant=parameters.exploration_constant,
            minimum_root_visits=parameters.minimum_root_visits,
        )
