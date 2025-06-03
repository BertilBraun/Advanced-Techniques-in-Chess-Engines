from __future__ import annotations
import random
import time

import chess
import numpy as np
from dataclasses import dataclass

from src.games.Board import Player
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.util import clamp, lerp
from src.mcts.MCTS import MCTS, action_probabilities
from src.mcts.MCTSNode import MCTSNode
from src.cluster.InferenceClient import InferenceClient
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import CURRENT_GAME, CurrentBoard, CurrentGame, CurrentGameMove, log_text, TRAINING_ARGS
from src.Encoding import get_board_result_score
from src.train.TrainingArgs import MCTSParams, SelfPlayParams
from src.util.log import log
from src.util.tensorboard import log_scalar, log_scalars
from src.util.timing import reset_times, timeit


@dataclass(frozen=True)
class SelfPlayGameMemory:
    board: CurrentBoard
    visit_counts: list[tuple[int, int]]
    result_score: float


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.played_moves: list[CurrentGameMove] = []
        self.already_expanded_node: MCTSNode | None = None
        self.start_generation_time = time.time()

        # The move at which the player resigned, if any. None if the game is still ongoing.
        self.resigned_at_move: int | None = None
        # The player who resigned, if any. None if the game is still ongoing.
        self.resignee: Player | None = None

    def expand(self, move: CurrentGameMove) -> SelfPlayGame:
        new_game = self.copy()
        new_game.board.make_move(move)
        new_game.played_moves.append(move)
        return new_game

    def copy(self) -> SelfPlayGame:
        game = SelfPlayGame()
        game.board = self.board.copy()
        game.memory = self.memory.copy()
        game.played_moves = self.played_moves.copy()
        game.start_generation_time = self.start_generation_time
        game.resigned_at_move = self.resigned_at_move
        game.resignee = self.resignee
        return game

    def __hash__(self) -> int:
        return self.board.quick_hash()


class SelfPlay:
    def __init__(self, client: InferenceClient, args: SelfPlayParams) -> None:
        self.client = client
        self.args = args

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts: MCTS | None = None

    def update_iteration(self, iteration: int) -> None:
        if len(self.dataset) > 0:
            log(f'Warning: Dataset should be empty when updating iteration. Discarding {len(self.dataset)} samples.')
        self.iteration = iteration
        self.dataset = SelfPlayDataset()
        self.client.update_iteration(iteration)

        self._set_mcts(iteration)

    def _set_mcts(self, iteration: int) -> None:
        """Set the MCTS parameters for the current iteration."""
        # start with 10% of the searches, scale up to 100% over the first 10% of total iterations
        num_searches_per_turn = int(
            lerp(
                self.args.mcts.num_searches_per_turn / 20,
                self.args.mcts.num_searches_per_turn,
                clamp(iteration * 20 / TRAINING_ARGS.num_iterations, 0.0, 1.0),
            )
        )
        assert (
            num_searches_per_turn > self.args.mcts.num_parallel_searches
        ), f'Number of searches per turn ({num_searches_per_turn}) must be greater than number of parallel searches ({self.args.mcts.num_parallel_searches}).'

        log_scalar('dataset/num_searches_per_turn', num_searches_per_turn, iteration)

        mcts_args = MCTSParams(
            num_searches_per_turn=num_searches_per_turn,
            num_parallel_searches=self.args.mcts.num_parallel_searches,
            dirichlet_alpha=self.args.mcts.dirichlet_alpha,
            dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
            c_param=self.args.mcts.c_param,
            min_visit_count=self.args.mcts.min_visit_count,
        )
        self.mcts = MCTS(self.client, mcts_args)

    def self_play(self) -> None:
        assert self.mcts is not None, 'MCTS must be set via update_iteration before self_play can be called.'

        mcts_results = self.mcts.search(
            [(spg.board, spg.already_expanded_node) for spg in self.self_play_games],
            should_run_full_search=[
                (
                    not self.args.only_store_sampled_moves  # If all moves are stored, run full search
                    or len(spg.played_moves)
                    < self.args.num_moves_after_which_to_play_greedy  # or if the game is still in the early phase
                    or len(spg.board.board.piece_map()) > 10  # or if there are many pieces on the board
                )
                and spg.resigned_at_move is None  # and the game has not been resigned
                for spg in self.self_play_games
            ],
        )

        for i, (spg, mcts_result) in enumerate(zip(self.self_play_games, mcts_results)):
            if mcts_result.is_full_search:
                spg.memory.append(
                    SelfPlayGameMemory(spg.board.copy(), mcts_result.visit_counts, mcts_result.result_score)
                )

            if mcts_result.result_score < self.args.resignation_threshold and spg.resigned_at_move is None:
                # Resignation if most of the mcts searches result in a loss
                self.dataset.stats += SelfPlayDatasetStats(resignations=1)

                if random.random() < 0.1:
                    # With 10% chance, play out the game to the end to see if it was winnable
                    spg.resigned_at_move = len(spg.played_moves)
                    spg.resignee = spg.board.current_player
                else:
                    self.self_play_games[i] = self._handle_end_of_game(spg, mcts_result.result_score)
                    continue

            if CURRENT_GAME == 'chess':
                if len(spg.played_moves) >= 250:
                    # If the game is too long, end it and add it to the dataset
                    self.dataset.stats += SelfPlayDatasetStats(num_too_long_games=1)
                    self.self_play_games[i] = self._handle_end_of_game(spg, 0.0)
                    continue

                pieces = list(spg.board.board.piece_map().values())
                white_pieces = sum(1 for piece in pieces if piece.color == chess.WHITE)
                black_pieces = sum(1 for piece in pieces if piece.color == chess.BLACK)
                if (
                    False
                    and (white_pieces < 4 or black_pieces < 4)
                    and len(spg.played_moves) >= 80
                    and random.random() < 0.2
                ):
                    # If there are only a few pieces left, and the game has been going on for a while, have a chance to end the game early and add it to the dataset to avoid noisy long games
                    from src.games.chess.ChessBoard import PIECE_VALUE

                    white_value = sum(PIECE_VALUE[piece.piece_type] for piece in pieces if piece.color == chess.WHITE)
                    black_value = sum(PIECE_VALUE[piece.piece_type] for piece in pieces if piece.color == chess.BLACK)

                    # Convert to result from current player's perspective
                    if spg.board.current_player == 1:  # White's perspective
                        if white_value > black_value:
                            game_outcome = 1.0
                        elif black_value > white_value:
                            game_outcome = -1.0
                        else:
                            game_outcome = 0.0
                    else:  # Black's perspective
                        if black_value > white_value:
                            game_outcome = 1.0
                        elif white_value > black_value:
                            game_outcome = -1.0
                        else:
                            game_outcome = 0.0

                    game_outcome *= 0.9  # somewhat unsure about the game outcome, therefore discount if with 0.9
                    self._add_training_data(spg, game_outcome)
                    self.self_play_games[i] = SelfPlayGame()
                    continue

            spg_action_probabilities = action_probabilities(mcts_result.visit_counts)

            while np.sum(spg_action_probabilities) > 0:
                new_spg, move = self._sample_self_play_game(spg, spg_action_probabilities, mcts_result.children)

                is_duplicate = any(hash(game) == hash(new_spg) for game in self.self_play_games)
                is_repetition = move in spg.played_moves[-8:]
                if is_duplicate or is_repetition:
                    # don't play the same move twice in a row
                    # Already exploring this state, so remove the probability of this move and try again
                    spg_action_probabilities[CurrentGame.encode_move(move, spg.board)] = 0
                else:
                    if new_spg.board.is_game_over():
                        # Game is over, add the game to the dataset
                        result = get_board_result_score(new_spg.board)
                        assert result is not None, 'Game should not be over if result is None'
                        self.self_play_games[i] = self._handle_end_of_game(new_spg, result)
                    else:
                        self.self_play_games[i] = new_spg
                    break

            else:
                # No valid moves left which are not already being explored
                # Therefore simply pick the most likely move, and expand to different states from the most likely next state in the next iteration
                new_spg, _ = self._sample_self_play_game(
                    spg, action_probabilities(mcts_result.visit_counts), mcts_result.children
                )
                if new_spg.board.is_game_over():
                    # Game is over, add the game to the dataset
                    result = get_board_result_score(new_spg.board)
                    assert result is not None, 'Game should not be over if result is None'
                    self.self_play_games[i] = self._handle_end_of_game(new_spg, result)
                else:
                    self.self_play_games[i] = new_spg

        reset_times()

    def _handle_end_of_game(self, spg: SelfPlayGame, game_outcome: float) -> SelfPlayGame:
        self._add_training_data(spg, game_outcome)

        if spg.resigned_at_move is not None:
            self.dataset.stats += SelfPlayDatasetStats(
                num_resignations_evaluated_to_end=1,
                num_winnable_resignations=1 if game_outcome > 0.5 and spg.resignee == spg.board.current_player else 0,
                num_moves_after_resignation=len(spg.played_moves) - spg.resigned_at_move,
            )

        return self._new_game()

    def _new_game(self) -> SelfPlayGame:
        # Create a new game instance
        new_game = SelfPlayGame()

        # Play a random moves to start the game in different states
        random_moves_to_play = int(random.random() * 5)
        for _ in range(random_moves_to_play):
            new_game = new_game.expand(random.choice(new_game.board.get_valid_moves()))

        return new_game

    @timeit
    def _sample_self_play_game(
        self, current: SelfPlayGame, action_probabilities: np.ndarray, children: list[MCTSNode]
    ) -> tuple[SelfPlayGame, CurrentGameMove]:
        # Sample a move from the action probabilities then create a new game state with that move
        # If the game is over, add the game to the dataset and return a new game state, thereby initializing a new game

        # only use temperature for the first X moves, then simply use the most likely move
        # Keep exploration high for the first X moves, then play out as well as possible to reduce noise in the backpropagated final game results
        if len(current.played_moves) >= self.args.num_moves_after_which_to_play_greedy:
            move = CurrentGame.decode_move(np.argmax(action_probabilities).item(), current.board)
        else:
            move = self._sample_move(action_probabilities, current.board, self.args.temperature)

        new_spg = current.expand(move)

        return new_spg, move
        # TODO encoded_move = CurrentGame.encode_move(move)
        # TODO new_spg.already_expanded_node = next(
        # TODO     child for child in children if child.encoded_move_to_get_here == encoded_move
        # TODO ).copy(parent=None)  # remove parent to avoid memory leaks

    def _sample_move(
        self, action_probabilities: np.ndarray, board: CurrentBoard, temperature: float = 1.0
    ) -> CurrentGameMove:
        assert temperature > 0, 'Temperature must be greater than 0'

        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)

        action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

        return CurrentGame.decode_move(action, board)

    @timeit
    def _add_training_data(self, spg: SelfPlayGame, game_outcome: float) -> None:
        # result: 1 if current player won, -1 if current player lost, 0 if draw

        self._log_game(spg, game_outcome)

        self.dataset.add_generation_stats(
            game_length=len(spg.played_moves),
            generation_time=time.time() - spg.start_generation_time,
        )

        for mem in spg.memory[::-1]:
            turn_game_outcome = game_outcome if mem.board.current_player == spg.board.current_player else -game_outcome

            for board, visit_counts in CurrentGame.symmetric_variations(mem.board, mem.visit_counts):
                self.dataset.add_sample(
                    board.astype(np.int8).copy(),
                    self._preprocess_visit_counts(visit_counts),
                    lerp(
                        turn_game_outcome,
                        mem.result_score,
                        clamp(self.iteration * 10 / TRAINING_ARGS.num_iterations, 0.0, 1.0)
                        * self.args.result_score_weight,
                    ),
                )

            game_outcome *= 0.99  # discount the game outcome for each move

    def _preprocess_visit_counts(self, visit_counts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # Remove moves which were only visited exactly as many times as required, never more
        visit_counts = [
            (move, count - self.args.mcts.min_visit_count)
            for move, count in visit_counts
            if count > self.args.mcts.min_visit_count
        ]

        return visit_counts

    def _log_game(self, spg: SelfPlayGame, result: float) -> None:
        moves: list[str] = []
        board = CurrentGame.get_initial_board()
        for move in spg.played_moves:
            encoded_move = CurrentGame.encode_move(move, board)
            moves.append(str(encoded_move))
            board.make_move(move)

        starting_line = moves[:5]  # first 5 moves
        starting_hash = sum(ord(c) * i for i, c in enumerate(''.join(starting_line)))
        log_scalars('self_play/starting_line', {str(starting_hash): 1}, self.iteration)
        log_text(f'starting_hash/{starting_hash}', ','.join(starting_line), self.iteration)

        if random.random() < 0.01:
            # log a game every 1% of the time
            moves_str = ','.join(moves)
            log_text(f'moves/{self.iteration}/{hash(moves_str)}', f'{result}:{moves_str}')
