import torch


from src.util.log import log
from src.alpha_zero.AlphaZero import AlphaZero
from src.Network import Network
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats


class ClusterAlphaZero:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingArgs,
        rank: int,
        load_latest_model: bool = True,
    ) -> None:
        assert args.cluster is not None, 'Cluster parameters must be set'

        self.az = AlphaZero(model, optimizer, args, load_latest_model)
        self.rank = rank
        self.trainers = args.cluster.num_train_nodes_on_cluster
        self.self_players = args.cluster.num_self_play_nodes_on_cluster

    def learn(self) -> None:
        if self.trainers == 0 and self.rank == 0:
            self._mix_self_play_and_train_on_cluster()
        else:
            if self.rank < self.trainers:
                self._train_on_cluster()
            else:
                self._self_play_on_cluster()

    def _mix_self_play_and_train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.az.starting_iteration

        for iteration in range(self.az.starting_iteration, self.az.args.num_iterations):
            num_self_play_calls = self.az.args.self_play.num_games_per_iteration // self.self_players // 2
            self.az._self_play_and_write_memory(iteration, num_self_play_calls)
            training_stats.append(self.az._train_and_save_new_model(iteration))
            self.az._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _self_play_on_cluster(self) -> None:
        # Starting iteration is always loaded in the _load_latest_model method
        while self.az.starting_iteration < self.az.args.num_iterations:
            num_self_play_calls = self.az.args.self_play.num_games_per_iteration // self.self_players
            self.az._self_play_and_write_memory(self.az.starting_iteration, num_self_play_calls)
            self.az._load_latest_model()

    def _train_on_cluster(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.az.starting_iteration

        for iteration in range(self.az.starting_iteration, self.az.args.num_iterations):
            training_stats.append(self.az._train_and_save_new_model(iteration))

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')
