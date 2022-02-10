from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Generic, List, TypeVar

import chess
import numpy as np
from tqdm import tqdm

MoveType = TypeVar('MoveType')

EvaluationFunction = Callable[['MCTSState', List[MoveType]], List[float]]


class MCTSNode(Generic[MoveType]):
    def __init__(self, state: MCTSState, evaluation: EvaluationFunction, move_to_get_here: MoveType, remaining_depth: int, parent: MCTSNode = None):
        self.state = state
        self.parent = parent
        self.evaluation = evaluation
        self.children = []
        self.move_to_get_here = move_to_get_here
        self.remaining_depth = remaining_depth
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions

    @property
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self) -> float:
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    @property
    def n(self) -> int:
        return self._number_of_visits

    @property
    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def expand(self) -> MCTSNode:
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MCTSNode(
            next_state,
            parent=self,
            move_to_get_here=action,
            remaining_depth=self.remaining_depth - 1,
            evaluation=self.evaluation
        )

        self.children.append(child_node)
        return child_node

    def rollout(self) -> int:
        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():

            possible_moves = current_rollout_state.get_legal_actions()
            evaluations = self.evaluation(
                current_rollout_state, possible_moves
            )
            action = possible_moves[np.argmax(evaluations)]
            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def backpropagate(self, result: int) -> None:
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def best_child(self, c_param: float = 0.1) -> MCTSNode:
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def _tree_policy(self) -> MCTSNode:
        current_node = self

        while not current_node.is_terminal_node and current_node.remaining_depth > 0:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        if current_node.remaining_depth <= 0:
            print("Warning: reached maximum depth")

        return current_node

    def best_action(self, iterations: int = 100) -> MoveType:
        for _ in tqdm(range(iterations)):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.).move_to_get_here


class MCTSState:
    @abstractmethod
    def get_legal_actions(self) -> List:
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible states from current state.
        Returns a list.
        '''
        return []

    @abstractmethod
    def is_game_over(self) -> bool:
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        return False

    @abstractmethod
    def game_result(self) -> int:
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return 0

    @abstractmethod
    def move(self, action: chess.Move) -> None:
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        pass


class ChessState(MCTSState):

    def __init__(self, state: chess.Board, turn: chess.Color):
        self.state = state
        self.turn = turn

    def get_legal_actions(self) -> List:
        return list(self.state.legal_moves)

    def is_game_over(self) -> bool:
        return self.state.is_game_over()

    def game_result(self) -> int:
        result = self.state.result()
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:
            return 0

    def move(self, action: chess.Move) -> MCTSState:
        new_state = self.state.copy(stack=False)
        new_state.push(action)
        return ChessState(new_state, not self.state.turn)


def mcts_player(board: chess.Board, evaluation: EvaluationFunction, iterations: int, maxdepth: int) -> chess.Move:
    for move_choice in board.legal_moves:
        copy = board.copy()
        copy.push(move_choice)
        if copy.is_game_over():
            board.push(move_choice)
            return move_choice

    root = MCTSNode(
        ChessState(board, board.turn),
        evaluation,
        move_to_get_here=None,
        remaining_depth=maxdepth
    )
    move = root.best_action(iterations)
    # move = UCT(board, itermax, depthmax, evaluation)
    board.push(move)
    return move


def mcts_player_with_stats(evaluation: EvaluationFunction, printAndResetStats: Callable, iterations: int = 500, maxdepth: int = 30):
    def inner(board: chess.Board):
        start = time.time()
        print("MCTS Player:", mcts_player(
            board, evaluation, iterations, maxdepth))
        print("Time:", time.time() - start)
        printAndResetStats()

    return inner
