import json
import torch
import tensorflow as tf
from tensorflow._api.v2.summary import create_file_writer

from tqdm import trange
from pathlib import Path

from src.alpha_zero.SelfPlay import SelfPlay, SelfPlayMemory
from src.eval.ModelEvaluation import ModelEvaluation
from src.util.log import log
from src.util import batched_iterate, load_json, random_id
from src.Network import Network, clear_model_inference_cache
from src.settings import CURRENT_GAME
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

    def learn(self) -> None:
        training_stats: list[TrainingStats] = []
        starting_iteration = self.starting_iteration

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            with create_file_writer(str(self.save_path / 'logs')).as_default():
                self._self_play_and_write_memory(
                    iteration,
                    self.args.num_self_play_games_per_iteration,
                )

                training_stats.append(self._train_and_save_new_model(iteration))

                self._load_latest_model()

        log('Training finished')
        log('Final training stats:')
        for i, stats in enumerate(training_stats):
            log(f'Iteration {starting_iteration + i + 1}: {stats}')

    def _self_play_and_write_memory(self, iteration: int, num_self_play_calls: int):
        memory: list[SelfPlayMemory] = []

        for _ in trange(
            num_self_play_calls // self.args.num_parallel_games,
            desc=f'Self Play for {self.args.num_parallel_games} games in parallel',
        ):
            memory += self.self_play.self_play(iteration)

        log(f'Collected {len(memory)} self-play memories.')
        self._save_memory(memory, iteration)

    def _train_and_save_new_model(self, iteration: int) -> TrainingStats:
        memory = self._load_all_memories_to_train_on_for_iteration(iteration)
        log(f'Loaded {len(memory)} self-play memories.')
        tf.summary.scalar('num_training_samples', len(memory), iteration)

        memory = self._deduplicate_positions(memory)
        log(f'Deduplicated to {len(memory)} unique positions.')
        tf.summary.scalar('num_deduplicated_samples', len(memory), iteration)

        tf.summary.histogram('training_sample_values', torch.tensor([mem.value_target for mem in memory]), iteration)

        # figure out average spikeyness of policy targets.
        # Should be close to 1 if the policy is very spikey, and close to 1/9 if it is very uniform.
        spikiness = sum((mem.policy_targets).max().item() for mem in memory) / len(memory)
        tf.summary.scalar(
            'policy_spikiness',
            spikiness,
            iteration,
            description='Close to 1 if spikey, close to 1/ACTION_SIZE if uniform',
        )
        tf.summary.histogram(
            'policy_targets', torch.stack([mem.policy_targets for mem in memory]).reshape(-1), iteration
        )

        train_stats = TrainingStats(self.args.batch_size)
        for epoch in range(self.args.num_epochs):
            epoch_train_stats = self.trainer.train(memory, iteration)
            tf.summary.scalar(
                'policy_loss',
                epoch_train_stats.policy_loss / epoch_train_stats.num_batches,
                iteration * self.args.num_epochs + epoch,
            )
            tf.summary.scalar(
                'value_loss',
                epoch_train_stats.value_loss / epoch_train_stats.num_batches,
                iteration * self.args.num_epochs + epoch,
            )
            tf.summary.scalar(
                'total_loss',
                epoch_train_stats.total_loss / epoch_train_stats.num_batches,
                iteration * self.args.num_epochs + epoch,
            )
            log(f'Epoch {epoch + 1}: {epoch_train_stats}')
            train_stats += epoch_train_stats

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
            model = Network(self.args.nn_num_layers, self.args.nn_hidden_size)
            model.load_state_dict(
                torch.load(self.save_path / f'model_{iteration}.pt', map_location=model.device, weights_only=True)
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

    def _save_memory(self, memory: list[SelfPlayMemory], iteration: int) -> None:
        memory_path = self.save_path / f'memory_{iteration}_{random_id()}.pt'
        torch.save(
            [(mem.state, mem.policy_targets, mem.value_target) for mem in memory],
            memory_path,
        )
        log(f'Memory saved at iteration {iteration}')

    def _load_all_memories_to_train_on_for_iteration(self, iteration: int) -> list[SelfPlayMemory]:
        window_size = self.args.sampling_window(iteration)

        memory: list[SelfPlayMemory] = []
        for iter in range(max(iteration - window_size, 0), iteration + 1):
            memory += self._load_all_memories(iter)
        return memory

    def _load_all_memories(self, iteration: int) -> list[SelfPlayMemory]:
        memory: list[SelfPlayMemory] = []

        for f in self.save_path.iterdir():
            if f.suffix == '.pt' and f.stem.startswith(f'memory_{iteration}_'):
                mapped_memory = torch.load(f, weights_only=True)
                memory += [SelfPlayMemory(*mem) for mem in mapped_memory]

        return memory

    def _deduplicate_positions(self, memory: list[SelfPlayMemory]) -> list[SelfPlayMemory]:
        """Deduplicate the positions in the memory by averaging the policy and value targets for the same board state."""
        mp: dict[int, tuple[int, SelfPlayMemory]] = {}
        for batch in batched_iterate(memory, 128):
            states = [mem.state for mem in batch]
            hashes = CURRENT_GAME.hash_boards(torch.stack(states))

            for mem, h in zip(batch, hashes):
                if h in mp:
                    count, spm = mp[h]
                    spm.policy_targets += mem.policy_targets
                    spm.value_target += mem.value_target
                    mp[h] = (count + 1, spm)
                else:
                    mp[h] = (1, mem)

        for count, spm in mp.values():
            spm.policy_targets /= count
            spm.value_target /= count

        return [spm for _, spm in mp.values()]

    def _play_two_most_recent_models(self, iteration: int) -> None:
        """Play two most recent models against each other."""

        current_model = self._load_model(iteration)
        previous_model = self._load_model(iteration - 1)

        model_evaluation = ModelEvaluation()
        results = model_evaluation.play_two_models_search(current_model, previous_model, 30)

        log(f'Results after playing two most recent models at iteration {iteration}:', results)

        tf.summary.scalar('win_loss_draw_vs_previous_model/wins', results.wins, iteration)
        tf.summary.scalar('win_loss_draw_vs_previous_model/losses', results.losses, iteration)
        tf.summary.scalar('win_loss_draw_vs_previous_model/draws', results.draws, iteration)

        results = model_evaluation.play_vs_random(current_model, 30)
        log(f'Results after playing vs random at iteration {iteration}:', results)

        tf.summary.scalar('win_loss_draw_vs_random/wins', results.wins, iteration)
        tf.summary.scalar('win_loss_draw_vs_random/losses', results.losses, iteration)
        tf.summary.scalar('win_loss_draw_vs_random/draws', results.draws, iteration)
