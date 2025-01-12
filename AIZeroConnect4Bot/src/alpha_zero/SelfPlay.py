import numpy as np
from dataclasses import dataclass

from viztracer import VizTracer

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CurrentBoard, CurrentGame, CurrentGameMove
from src.Encoding import get_board_result_score
from src.alpha_zero.train.TrainingArgs import SelfPlayParams
from src.util import lerp
from src.util.log import log


@dataclass
class SelfPlayGameMemory:
    board: CurrentBoard
    action_probabilities: np.ndarray
    result_score: float


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CurrentGame.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.num_played_moves = 0


def sample_move(
    action_probabilities: np.ndarray, num_played_moves: int = 0, temperature: float = 1.0
) -> CurrentGameMove:
    # only use temperature for the first 30 moves, then simply use the action probabilities as they are
    if num_played_moves < 30:
        temperature_action_probabilities = action_probabilities ** (1 / temperature)
        temperature_action_probabilities /= np.sum(temperature_action_probabilities)
    else:
        temperature_action_probabilities = action_probabilities

    action = np.random.choice(CurrentGame.action_size, p=temperature_action_probabilities)

    return CurrentGame.decode_move(action)


class SelfPlay:
    def __init__(self, client: InferenceClient, args: SelfPlayParams) -> None:
        self.client = client
        self.args = args

        self.self_play_games: list[SelfPlayGame] = [SelfPlayGame() for _ in range(self.args.num_parallel_games)]
        self.dataset = SelfPlayDataset()

        self.iteration = 0

        self.mcts = self._get_mcts(self.iteration)

        self.tracer = VizTracer(tracer_entries=10_000_000, max_stack_depth=11)
        self.tracer.start()

    def update_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self.mcts = self._get_mcts(self.iteration)
        self.dataset = SelfPlayDataset()
        self.client.update_iteration(iteration)

    async def self_play(self) -> None:
        log('Self play move:', self.self_play_games[0].num_played_moves)
        mcts_results = await self.mcts.search([spg.board for spg in self.self_play_games])

        for spg, (action_probabilities, result_score) in zip(self.self_play_games, mcts_results):
            spg.memory.append(SelfPlayGameMemory(spg.board.copy(), action_probabilities, result_score))

            move = sample_move(action_probabilities, spg.num_played_moves, self.args.temperature)
            spg.board.make_move(move)
            spg.num_played_moves += 1

            if spg.board.is_game_over():
                self.dataset += self._get_training_data(spg)

        self.self_play_games = [spg for spg in self.self_play_games if not spg.board.is_game_over()]
        num_games_to_restart = self.args.num_parallel_games - len(self.self_play_games)
        self.self_play_games += [SelfPlayGame() for _ in range(num_games_to_restart)]

        self.tracer.stop()
        self.tracer.save(f'self_play_{spg.num_played_moves}.json')
        self.tracer = VizTracer(tracer_entries=10_000_000, max_stack_depth=11)
        self.tracer.start()
        if spg.num_played_moves == 5:
            exit()

    def _get_mcts(self, iteration: int) -> MCTS:
        return MCTS(
            self.client,
            MCTSArgs(
                num_searches_per_turn=self.args.mcts.num_searches_per_turn,
                num_parallel_searches=self.args.mcts.num_parallel_searches,
                dirichlet_epsilon=self.args.mcts.dirichlet_epsilon,
                dirichlet_alpha=self.args.mcts.dirichlet_alpha(iteration),
                c_param=self.args.mcts.c_param,
            ),
        )

    def _get_training_data(self, spg: SelfPlayGame) -> SelfPlayDataset:
        self_play_dataset = SelfPlayDataset()

        # 1 if current player won, -1 if current player lost, 0 if draw
        result = get_board_result_score(spg.board)
        assert result is not None, 'Game is not over'

        for mem in spg.memory[::-1]:  # reverse to flip the result for the other player
            encoded_board = CurrentGame.get_canonical_board(mem.board)

            for board, probabilities in CurrentGame.symmetric_variations(encoded_board, mem.action_probabilities):
                self_play_dataset.add_sample(
                    board.copy().astype(np.int8),
                    probabilities.copy().astype(np.float32),
                    lerp(result, mem.result_score, self.args.result_score_weight),
                )
            result = -result

        return self_play_dataset
