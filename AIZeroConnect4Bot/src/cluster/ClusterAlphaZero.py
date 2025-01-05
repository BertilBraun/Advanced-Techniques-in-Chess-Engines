import time
import torch


from src.util.log import log
from src.alpha_zero.AlphaZero import AlphaZero
from src.Network import Network
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.settings import CurrentGame


class ClusterAlphaZero:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingArgs,
        load_latest_model: bool = True,
    ) -> None:
        self.az = AlphaZero(model, optimizer, args, load_latest_model)

        assert args.cluster is not None, 'Cluster parameters must be set'
        self.self_players = args.cluster.num_self_play_nodes_on_cluster

    def mix_self_play_and_train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.az.starting_iteration

        for iteration in range(self.az.starting_iteration, self.az.args.num_iterations):
            num_self_play_calls = self.az.args.self_play.num_games_per_iteration // self.self_players // 2
            self.az._self_play_and_write_memory(iteration, num_self_play_calls)
            training_stats.append(self._train_one_iteration(iteration))
            self.az._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def self_play_on_cluster(self) -> None:
        # Starting iteration is always loaded in the _load_latest_model method
        while self.az.starting_iteration < self.az.args.num_iterations:
            num_self_play_calls = self.az.args.self_play.num_games_per_iteration // self.self_players
            self.az._self_play_and_write_memory(self.az.starting_iteration, num_self_play_calls)
            self.az._load_latest_model()

    def train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.az.starting_iteration

        for iteration in range(self.az.starting_iteration, self.az.args.num_iterations):
            training_stats.append(self._train_one_iteration(iteration))

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _train_one_iteration(self, iteration: int) -> TrainingStats:
        EXPECTED_NUM_SAMPLES = self.az.args.self_play.num_games_per_iteration * CurrentGame.average_num_moves_per_game

        while (
            len(SelfPlayDataset.load_iteration(self.az.args.save_path, iteration, device=torch.device('cpu')))
            < EXPECTED_NUM_SAMPLES
        ):
            log('Waiting for memories...')
            time.sleep(20)

        return self.az._train_and_save_new_model(iteration)
