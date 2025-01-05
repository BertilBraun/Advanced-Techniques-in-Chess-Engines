import json
import time
from typing import Generator
import numpy as np
import torch

from tqdm import trange
from pathlib import Path

from src.alpha_zero.SelfPlay import SelfPlay
from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.eval.ModelEvaluation import ModelEvaluation
from src.settings import DEDUPLICATE_EACH_ITERATION, log_histogram, log_scalar
from src.util.compile import try_compile
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util import load_json
from src.Network import Network, clear_model_inference_cache
from src.alpha_zero.train.Trainer import Trainer
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.alpha_zero.train.TrainingStats import TrainingStats


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

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            self._self_play_and_write_memory(
                iteration,
                self.args.self_play.num_games_per_iteration,
            )

            training_stats.append(self._train_and_save_new_model(iteration))
            yield iteration, training_stats[-1]

            self._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _self_play_and_write_memory(self, iteration: int, num_self_play_calls: int):
        dataset = SelfPlayDataset(self.model.device)

        for _ in trange(
            num_self_play_calls // self.args.self_play.num_parallel_games,
            desc=f'Self Play for {self.args.self_play.num_parallel_games} games in parallel',
        ):
            with log_exceptions('Self play'):
                dataset += self.self_play.self_play(iteration)

        log(f'Collected {len(dataset)} self-play memories.')
        dataset.save(self.save_path, iteration)

    def _train_and_save_new_model(self, iteration: int) -> TrainingStats:
        train_stats = TrainingStats()

        with log_exceptions('Training'):
            dataset = self._load_all_memories_to_train_on_for_iteration(iteration)
            while len(dataset) == 0:
                log('No memories found, waiting for self-play to finish...')
                time.sleep(5)
                dataset = self._load_all_memories_to_train_on_for_iteration(iteration)
            log(f'Loaded {len(dataset)} self-play memories.')

            log_scalar('num_training_samples', len(dataset), iteration)

            if not DEDUPLICATE_EACH_ITERATION:
                dataset.deduplicate()
                log(f'Deduplicated to {len(dataset)} unique positions.')

            log_scalar('num_deduplicated_samples', len(dataset), iteration)
            log_histogram('training_sample_states', np.array(dataset.states), iteration)

            # The spikiness of the policy targets.
            # The more confident the policy is, the closer to 1 it will be. I.e. the policy is sure about the best move.
            spikiness = dataset.policy_targets.max(axis=1).mean().item()
            log_scalar('policy_spikiness', spikiness, iteration)

            log_histogram('policy_targets', dataset.policy_targets, iteration)

            for epoch in range(self.args.training.num_epochs):
                epoch_train_stats = self.trainer.train(dataset, iteration)
                log_scalar(
                    'policy_loss',
                    epoch_train_stats.policy_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                log_scalar(
                    'value_loss',
                    epoch_train_stats.value_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                log_scalar(
                    'total_loss',
                    epoch_train_stats.total_loss / epoch_train_stats.num_batches,
                    iteration * self.args.training.num_epochs + epoch,
                )
                log(f'Epoch {epoch + 1}: {epoch_train_stats}')
                train_stats += epoch_train_stats

            log(f'Iteration {iteration + 1}: {train_stats}')
            self._save_latest_model(iteration)

            self._play_two_most_recent_models(iteration)

        return train_stats

    def _load_latest_model(self) -> None:
        """Load the latest model and optimizer from the last_training_config.json file if it exists, otherwise start from scratch."""
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
            model = try_compile(model)
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

        log(
            f'Loading memories for iteration {iteration} with window size {window_size} ({max(iteration - window_size, 0)}-{iteration})'
        )

        dataset = SelfPlayDataset(self.model.device)
        for iter in range(max(iteration - window_size, 0), iteration + 1):
            iteration_dataset = SelfPlayDataset.load_iteration(self.save_path, iter, self.model.device)
            file_paths = SelfPlayDataset.get_files_to_load_for_iteration(self.save_path, iter)

            if len(iteration_dataset) != 0 and len(file_paths) > 1 and DEDUPLICATE_EACH_ITERATION:
                iteration_dataset.deduplicate()
                for file_path in file_paths:
                    Path(file_path).unlink()
                iteration_dataset.save(self.save_path, iter, 'deduplicated')
                log(f'Deduplicated memory_{iter} to {len(iteration_dataset)}')

            dataset += iteration_dataset

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

        log_scalar('win_loss_draw_vs_previous_model/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_previous_model/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_previous_model/draws', results.draws, iteration)

        results = model_evaluation.play_vs_random(
            current_model, self.args.evaluation.num_games, self.args.evaluation.num_searches_per_turn
        )
        log(f'Results after playing vs random at iteration {iteration}:', results)

        log_scalar('win_loss_draw_vs_random/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_random/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_random/draws', results.draws, iteration)
