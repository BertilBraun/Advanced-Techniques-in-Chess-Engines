import json
from typing import Generator
import torch
import tensorflow as tf
from tensorflow._api.v2.summary import create_file_writer

from tqdm import trange
from pathlib import Path

from src.alpha_zero.SelfPlay import SelfPlay
from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.eval.ModelEvaluation import ModelEvaluation
from src.util.log import log
from src.util import load_json, random_id
from src.Network import Network, clear_model_inference_cache
from src.alpha_zero.train.Trainer import Trainer
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats
from src.util.profiler import log_event


def print_mem():
    device = torch.device('cuda')
    print(torch.cuda.memory.mem_get_info(device))
    print(torch.cuda.memory_summary(device))
    print(torch.cuda.memory_reserved(device))
    print(torch.cuda.memory_allocated(device))


class AlphaZero:
    def __init__(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        args: TrainingArgs,
        load_latest_model: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.starting_iteration = 0
        self.self_play = SelfPlay(model, args)
        self.trainer = Trainer(model, optimizer, args)

        self.save_path = Path(self.args.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        if load_latest_model:
            self._load_latest_model()

    def learn(self) -> Generator[tuple[int, TrainingStats], None, None]:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.starting_iteration

        print('Memory pre learn')
        print_mem()

        with create_file_writer(str(self.save_path / 'logs')).as_default():
            for iteration in range(self.starting_iteration, self.args.num_iterations):
                print('Memory pre iteration')
                print_mem()
                self._self_play_and_write_memory(
                    iteration,
                    self.args.self_play.num_games_per_iteration,
                )
                print('Memory post self play')
                print_mem()

                training_stats.append(self._train_and_save_new_model(iteration))
                yield iteration, training_stats[-1]

                self._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _self_play_and_write_memory(self, iteration: int, num_self_play_calls: int):
        dataset = SelfPlayDataset()

        with log_event('self_play'):
            for _ in trange(
                num_self_play_calls // self.args.self_play.num_parallel_games,
                desc=f'Self Play for {self.args.self_play.num_parallel_games} games in parallel',
            ):
                dataset += self.self_play.self_play(iteration)

        log(f'Collected {len(dataset)} self-play memories.')
        dataset.save(self.save_path / f'memory_{iteration}_{random_id()}.pt')

    def _train_and_save_new_model(self, iteration: int) -> TrainingStats:
        with log_event('dataset_loading'):
            print('Pre dataset loading')
            print_mem()
            dataset = self._load_all_memories_to_train_on_for_iteration(iteration)
            print('Post dataset loading')
            print_mem()

            log(f'Loaded {len(dataset)} self-play memories.')
            tf.summary.scalar('num_training_samples', len(dataset), iteration)

        with log_event('dataset_deduplication'):
            dataset.deduplicate()
            log(f'Deduplicated to {len(dataset)} unique positions.')

        torch.cuda.memory._dump_snapshot('my_snapshot.pickle')

        with log_event('dataset_logging'):
            tf.summary.scalar('num_deduplicated_samples', len(dataset), iteration)

            tf.summary.histogram('training_sample_values', dataset.value_targets.cpu(), iteration)

            spikiness = sum((policy_targets).max().item() for policy_targets in dataset.policy_targets) / len(dataset)
            tf.summary.scalar(
                'policy_spikiness',
                spikiness,
                iteration,
                description="""The spikiness of the policy targets.
    The more confident the policy is, the closer to 1 it will be. I.e. the policy is sure about the best move.
    The more uniform the policy is, the closer to 1/ACTION_SIZE it will be. I.e. the policy is unsure about the best move.""",
            )
            tf.summary.histogram('policy_targets', dataset.policy_targets.reshape(-1).cpu(), iteration)

        with log_event('training'):
            train_stats = TrainingStats()
            for epoch in range(self.args.training.num_epochs):
                epoch_train_stats = self.trainer.train(dataset, iteration)
                tf.summary.scalar(
                    'policy_loss',
                    epoch_train_stats.policy_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                tf.summary.scalar(
                    'value_loss',
                    epoch_train_stats.value_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                tf.summary.scalar(
                    'total_loss',
                    epoch_train_stats.total_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                log(f'Epoch {epoch + 1}: {epoch_train_stats}')
                train_stats += epoch_train_stats

            del dataset
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            log(f'Iteration {iteration + 1}: {train_stats}')
            self._save_latest_model(iteration)

            if iteration > 0:
                self._play_two_most_recent_models(iteration)

            return train_stats

    def _load_latest_model(self) -> None:
        """Load the latest model and optimizer from the last_training_config.pt file if it exists, otherwise start from scratch."""
        try:
            last_training_config = load_json(self.save_path / 'last_training_config.json')

            new_starting_iteration = int(last_training_config['iteration']) + 1
            if self.starting_iteration == new_starting_iteration:
                log(f'No new model found, starting from iteration {self.starting_iteration}')
                return

            clear_model_inference_cache(self.starting_iteration)

            self.starting_iteration = new_starting_iteration
            self.model.load_state_dict(
                torch.load(last_training_config['model'], map_location=self.model.device, weights_only=True)
            )
            self.optimizer.load_state_dict(
                torch.load(last_training_config['optimizer'], map_location=self.model.device, weights_only=True)
            )

            log(f'Model and optimizer loaded from iteration {self.starting_iteration}')
        except FileNotFoundError:
            log('No model and optimizer found, starting from scratch')

    def _load_model(self, iteration: int) -> Network:
        try:
            model = Network(self.args.network.num_layers, self.args.network.hidden_size, self.model.device)
            model.load_state_dict(
                torch.load(self.save_path / f'model_{iteration}.pt', map_location=self.model.device, weights_only=True)
            )
            return model
        except FileNotFoundError:
            log(f'No model found for iteration {iteration}')
            raise

    def _save_latest_model(self, iteration: int) -> None:
        """Save the model and optimizer to the current directory with the current iteration number. Also save the current training configuration to last_training_config.json."""

        model_path = self.save_path / f'model_{iteration}.pt'
        optimizer_path = self.save_path / f'optimizer_{iteration}.pt'
        last_training_config_path = self.save_path / 'last_training_config.json'

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        with open(last_training_config_path, 'w') as f:
            json.dump(
                {
                    'model': str(model_path),
                    'optimizer': str(optimizer_path),
                    'iteration': iteration,
                },
                f,
                indent=4,
            )

        log(f'Model and optimizer saved at iteration {iteration}')

    def _load_all_memories_to_train_on_for_iteration(self, iteration: int) -> SelfPlayDataset:
        window_size = self.args.training.sampling_window(iteration)

        dataset = SelfPlayDataset()
        for iter in range(max(iteration - window_size, 0), iteration + 1):
            dataset += SelfPlayDataset.load_iteration(self.save_path, iter, self.model.device)

        return dataset

    def _play_two_most_recent_models(self, iteration: int) -> None:
        """Play two most recent models against each other."""
        if not self.args.evaluation or iteration % self.args.evaluation.every_n_iterations != 0 or iteration == 0:
            return

        current_model = self._load_model(iteration)
        previous_model = self._load_model(iteration - self.args.evaluation.every_n_iterations)

        model_evaluation = ModelEvaluation()
        results = model_evaluation.play_two_models_search(
            current_model, previous_model, self.args.evaluation.num_games, self.args.evaluation.num_searches_per_turn
        )

        log(f'Results after playing two most recent models at iteration {iteration}:', results)

        tf.summary.scalar('win_loss_draw_vs_previous_model/wins', results.wins, iteration)
        tf.summary.scalar('win_loss_draw_vs_previous_model/losses', results.losses, iteration)
        tf.summary.scalar('win_loss_draw_vs_previous_model/draws', results.draws, iteration)

        results = model_evaluation.play_vs_random(
            current_model, self.args.evaluation.num_games, self.args.evaluation.num_searches_per_turn
        )
        log(f'Results after playing vs random at iteration {iteration}:', results)

        tf.summary.scalar('win_loss_draw_vs_random/wins', results.wins, iteration)
        tf.summary.scalar('win_loss_draw_vs_random/losses', results.losses, iteration)
        tf.summary.scalar('win_loss_draw_vs_random/draws', results.draws, iteration)
