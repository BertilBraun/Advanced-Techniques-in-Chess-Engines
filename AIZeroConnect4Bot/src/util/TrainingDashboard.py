import json
from os import PathLike
import threading

import tensorflow as tf


class TrainingDashboard:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TrainingDashboard, cls).__new__(cls)
                    cls._instance.reset_iteration_data()
        return cls._instance

    def reset_iteration_data(self):
        self.iteration_data = {
            'losses': {'policy': 0.0, 'value': 0.0, 'total': 0.0},
            'num_training_samples': 0,
            'num_deduplicated_samples': 0,
            'cache_hit_rate': 0.0,
            'unique_positions_in_cache': 0,
            'nn_output_value_distribution': [],
            'training_sample_values': [],
            'policy_spikiness': 0.0,
            'learning_rate': 0.0,
            'win_loss_draw_vs_previous_model': {'wins': 0, 'losses': 0, 'draws': 0},
            'win_loss_draw_vs_random': {'wins': 0, 'losses': 0, 'draws': 0},
        }

    def log_policy_loss(self, value: float):
        tf.summary.scalar('policy_loss', value, step=0)
        self.iteration_data['losses']['policy'] = value

    def log_value_loss(self, value: float):
        self.iteration_data['losses']['value'] = value

    def log_total_loss(self, value: float):
        self.iteration_data['losses']['total'] = value

    def log_num_training_samples(self, count: int):
        self.iteration_data['num_training_samples'] = count

    def log_num_deduplicated_samples(self, count: int):
        self.iteration_data['num_deduplicated_samples'] = count

    def log_cache_hit_rate(self, hit_rate: float):
        self.iteration_data['cache_hit_rate'] = hit_rate

    def log_unique_positions_in_cache(self, count: int):
        self.iteration_data['unique_positions_in_cache'] = count

    def log_nn_output_value_distribution(self, distribution: list[float]):
        self.iteration_data['nn_output_value_distribution'] = distribution

    def log_training_sample_values(self, values: list[float]):
        self.iteration_data['training_sample_values'] = values

    def log_policy_spikiness(self, spikiness: float):
        self.iteration_data['policy_spikiness'] = spikiness

    def log_learning_rate(self, lr: float):
        self.iteration_data['learning_rate'] = lr

    def log_win_loss_draw_vs_previous_model(self, wins: int, losses: int, draws: int):
        self.iteration_data['win_loss_draw_vs_previous_model'] = {'wins': wins, 'losses': losses, 'draws': draws}

    def log_win_loss_draw_vs_random(self, wins: int, losses: int, draws: int):
        self.iteration_data['win_loss_draw_vs_random'] = {'wins': wins, 'losses': losses, 'draws': draws}

    def dump_iteration_data(self, save_path: str | PathLike, iteration: int):
        filepath = f'{save_path}/training_iteration_{iteration}.json'
        with open(filepath, 'w') as f:
            json.dump(self.iteration_data, f, indent=4)
        self.reset_iteration_data()

    def _load_all_iteration_data(self, save_path: str) -> list[dict]:
        iteration_data = []
        for i in range(1000):
            try:
                with open(f'{save_path}/training_iteration_{i}.json', 'r') as f:
                    iteration_data.append(json.load(f))
            except FileNotFoundError:
                break
        return iteration_data

    def display(self, save_path: str):
        iteration_data = self._load_all_iteration_data(save_path)
        # Display iteration data in a dashboard
        from matplotlib import pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.suptitle('Training Dashboard')
        plt.subplot(2, 3, 1)
        plt.title('Losses')
        plt.plot([data['losses']['total'] for data in iteration_data], label='Total loss')
        plt.plot([data['losses']['policy'] for data in iteration_data], label='Policy loss')
        plt.plot([data['losses']['value'] for data in iteration_data], label='Value loss')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.title('Training samples')
        plt.plot([data['num_training_samples'] for data in iteration_data], label='Num training samples')
        plt.plot([data['num_deduplicated_samples'] for data in iteration_data], label='Num deduplicated samples')
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.title('Cache')
        plt.plot(
            [data['cache_hit_rate'] * data['unique_positions_in_cache'] for data in iteration_data],
            label='Cache hit rate',
        )
        plt.plot([data['unique_positions_in_cache'] for data in iteration_data], label='Unique positions in cache')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.title('NN output value distribution')
        for data in iteration_data:
            plt.plot(list(sorted(data['nn_output_value_distribution'])))

        plt.subplot(2, 3, 5)
        plt.title('Training sample values')
        for data in iteration_data:
            plt.plot(list(sorted(data['training_sample_values'])))

        plt.subplot(2, 3, 6)
        plt.title('Win/Loss/Draw')
        plt.plot(
            [data['win_loss_draw_vs_previous_model']['wins'] for data in iteration_data], label='Wins vs previous model'
        )
        plt.plot(
            [data['win_loss_draw_vs_previous_model']['losses'] for data in iteration_data],
            label='Losses vs previous model',
        )
        plt.plot(
            [data['win_loss_draw_vs_previous_model']['draws'] for data in iteration_data],
            label='Draws vs previous model',
        )
        plt.plot([data['win_loss_draw_vs_random']['wins'] for data in iteration_data], label='Wins vs random')
        plt.plot([data['win_loss_draw_vs_random']['losses'] for data in iteration_data], label='Losses vs random')
        plt.plot([data['win_loss_draw_vs_random']['draws'] for data in iteration_data], label='Draws vs random')
        plt.legend()

        plt.show()


# Usage
# dashboard = TrainingDashboard()
# dashboard.log_policy_loss(0.5)
# dashboard.log_value_loss(0.2)
# dashboard.log_total_loss(0.7)
# ...
# dashboard.dump_iteration_data('path/to/save', 0)
# dashboard.display('path/to/save')
