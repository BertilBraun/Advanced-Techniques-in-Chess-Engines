import time
import torch

from torch.optim import Adam

from AIZeroConnect4Bot.src.settings import CURRENT_GAME, TORCH_DTYPE
from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.cluster.ClusterManager import ClusterManager
from AIZeroConnect4Bot.src.Network import Network
from AIZeroConnect4Bot.src.train.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.train.TrainingStats import TrainingStats


class ClusterAlphaZero(AlphaZero):
    def __init__(self, args: TrainingArgs, load_latest_model: bool = True) -> None:
        assert args.num_self_play_nodes_on_cluster is not None
        assert args.num_train_nodes_on_cluster is not None
        self.self_players = args.num_self_play_nodes_on_cluster
        self.trainers = args.num_train_nodes_on_cluster
        assert self.trainers in [0, 1], 'For now, only one trainer is supported'

        self.cluster_manager = ClusterManager(self.self_players + self.trainers)
        self.cluster_manager.initialize()

        model = Network()
        # move model to rank device
        model = model.to(
            device=torch.device('cuda', self.cluster_manager.rank),
            dtype=TORCH_DTYPE,
            non_blocking=False,
        )

        torch.set_float32_matmul_precision('high')
        # if torch.cuda.is_available():
        #    model: Network = torch.compile(model)  # type: ignore

        optimizer = Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

        super().__init__(model, optimizer, args, load_latest_model)

    def learn(self) -> None:
        if self.trainers == 0 and self.cluster_manager.is_root_node:
            self._mix_self_play_and_train_on_cluster()
        else:
            if not self.cluster_manager.is_root_node:
                time.sleep(60)  # wait for root node to start - especially compiling the model and hash functions
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

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {i + 1}: {stats}')

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

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {i + 1}: {stats}')

    def _train_one_iteration(self, iteration: int) -> TrainingStats:
        EXPECTED_NUM_SAMPLES = self.args.num_self_play_iterations * CURRENT_GAME.average_num_moves_per_game

        while len(self._load_all_memories(iteration)) < EXPECTED_NUM_SAMPLES:
            log('Waiting for memories...')
            time.sleep(60)

        return self._train_and_save_new_model(iteration)
