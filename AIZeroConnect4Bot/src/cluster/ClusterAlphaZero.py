import torch

from torch.optim import AdamW

from src.settings import USE_GPU
from src.util.compile import try_compile
from src.util.log import log
from src.alpha_zero.AlphaZero import AlphaZero
from src.cluster.ClusterManager import ClusterManager
from src.Network import Network
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.util.profiler import start_usage_logger


class ClusterAlphaZero(AlphaZero):
    def __init__(self, args: TrainingArgs, load_latest_model: bool = True) -> None:
        assert args.cluster, 'ClusterAlphaZero should only be used with cluster training'
        self.self_players = args.cluster.num_self_play_nodes_on_cluster
        self.trainers = args.cluster.num_train_nodes_on_cluster
        assert self.trainers in [0, 1], 'For now, only one trainer is supported'

        self.cluster_manager = ClusterManager(self.self_players + self.trainers)
        self.cluster_manager.initialize()

        if self.cluster_manager.is_root_node:
            start_usage_logger()

        # move model to rank device
        if USE_GPU:
            node_device = torch.device('cuda', self.cluster_manager.rank % torch.cuda.device_count())
        else:
            node_device = torch.device('cpu')

        model = Network(args.network.num_layers, args.network.hidden_size, device=node_device)
        model = try_compile(model)

        optimizer = AdamW(model.parameters(), lr=0.2, weight_decay=1e-4)

        super().__init__(model, optimizer, args, load_latest_model)

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
        starting_iteration = self.starting_iteration

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            num_self_play_calls = self.args.self_play.num_games_per_iteration // self.self_players // 2
            self._self_play_and_write_memory(iteration, num_self_play_calls)
            training_stats.append(self._train_and_save_new_model(iteration))
            self._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _self_play_on_cluster(self) -> None:
        # Starting iteration is always loaded in the _load_latest_model method
        while self.starting_iteration < self.args.num_iterations:
            num_self_play_calls = self.args.self_play.num_games_per_iteration // self.self_players
            self._self_play_and_write_memory(self.starting_iteration, num_self_play_calls)
            self._load_latest_model()

    def _train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.starting_iteration

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            training_stats.append(self._train_and_save_new_model(iteration))

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')
