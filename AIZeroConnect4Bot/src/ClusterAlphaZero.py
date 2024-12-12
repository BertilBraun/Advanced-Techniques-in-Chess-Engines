import time
import torch


from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.ClusterManager import ClusterManager
from AIZeroConnect4Bot.src.Network import Network
from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.TrainingStats import TrainingStats
from AIZeroConnect4Bot.src.settings import AVERAGE_NUM_MOVES_PER_GAME


class ClusterAlphaZero(AlphaZero):
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingArgs,
        load_latest_model: bool = True,
    ) -> None:
        super().__init__(model, optimizer, args, load_latest_model)

        assert args.num_self_play_nodes_on_cluster is not None
        assert args.num_train_nodes_on_cluster is not None
        self.self_players = args.num_self_play_nodes_on_cluster
        self.trainers = args.num_train_nodes_on_cluster
        assert self.trainers in [0, 1], 'For now, only one trainer is supported'

        self.cluster_manager = ClusterManager(self.self_players + self.trainers)
        self.cluster_manager.initialize()

    def learn(self) -> None:
        if self.trainers == 0 and self.cluster_manager.is_root_node:
            self._mix_self_play_and_train_on_cluster()
        else:
            if self.cluster_manager.rank < self.trainers:
                self._train_on_cluster()
            else:
                self._self_play_on_cluster()

    def _mix_self_play_and_train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            num_self_play_calls = self.args.num_self_play_iterations // self.self_players
            self._self_play_and_write_memory(iteration, num_self_play_calls)
            training_stats.append(self._train_and_save_new_model(iteration))
            self._load_latest_model()

        print('Training finished')
        print('Final training stats:')
        for i, stats in enumerate(training_stats):
            print(f'Iteration {i + 1}: {stats}')

    def _self_play_on_cluster(self) -> None:
        # Starting iteration is always loaded from the latest model
        while self.starting_iteration < self.args.num_iterations:
            num_self_play_calls = self.args.num_self_play_iterations // self.self_players
            self._self_play_and_write_memory(self.starting_iteration, num_self_play_calls)
            self._load_latest_model()

    def _train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            training_stats.append(self._train_one_iteration(iteration))

        print('Training finished')
        print('Final training stats:')
        for i, stats in enumerate(training_stats):
            print(f'Iteration {i + 1}: {stats}')

    def _train_one_iteration(self, iteration: int) -> TrainingStats:
        EXPECTED_NUM_SAMPLES = self.args.num_self_play_iterations * AVERAGE_NUM_MOVES_PER_GAME

        while len(self._load_all_memories(iteration)) < EXPECTED_NUM_SAMPLES:
            print('Waiting for memories...')
            time.sleep(60)

        return self._train_and_save_new_model(iteration)
