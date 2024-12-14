import json
import torch

from tqdm import trange
from pathlib import Path

from AIZeroConnect4Bot.src.util import batched_iterate, random_id
from AIZeroConnect4Bot.src.Network import Network, clear_cache
from AIZeroConnect4Bot.src.settings import CURRENT_GAME
from AIZeroConnect4Bot.src.train.Training import Trainer
from AIZeroConnect4Bot.src.train.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.train.TrainingStats import TrainingStats
from AIZeroConnect4Bot.src.self_play.SelfPlay import SelfPlay, SelfPlayMemory


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
        self.save_path.mkdir(exist_ok=True)

        if load_latest_model:
            self._load_latest_model()

    def learn(self) -> None:
        training_stats: list[TrainingStats] = []

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            self._self_play_and_write_memory(
                iteration,
                self.args.num_self_play_iterations,
            )

            training_stats.append(self._train_and_save_new_model(iteration))

            self._load_latest_model()

        print('Training finished')
        print('Final training stats:')
        for i, stats in enumerate(training_stats):
            print(f'Iteration {i + 1}: {stats}')

    def _self_play_and_write_memory(self, iteration: int, num_self_play_calls: int):
        memory: list[SelfPlayMemory] = []

        for _ in trange(
            num_self_play_calls // self.args.num_parallel_games,
            desc=f'Self Play for {self.args.num_parallel_games} games in parallel',
        ):
            memory += self.self_play.self_play()

        print(f'Collected {len(memory)} self-play memories.')
        self._save_memory(memory, iteration)

    def _train_and_save_new_model(self, iteration: int) -> TrainingStats:
        memory = self._load_all_memories_to_train_on_for_iteration(iteration)
        print(f'Loaded {len(memory)} self-play memories.')
        memory = self._deduplicate_positions(memory)
        print(f'Deduplicated to {len(memory)} unique positions.')

        train_stats = TrainingStats()
        for epoch in range(self.args.num_epochs):
            train_stats += self.trainer.train(memory, iteration)
            print(f'Epoch {epoch + 1}: {train_stats}')

        print(f'Iteration {iteration + 1}: {train_stats}')
        self._save_latest_model(iteration)
        return train_stats

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
                    spm.value_targets += mem.value_targets
                    mp[h] = (count + 1, spm)
                else:
                    mp[h] = (1, mem)

        for count, spm in mp.values():
            spm.policy_targets /= count
            spm.value_targets /= count

        return [spm for _, spm in mp.values()]

    def _load_latest_model(self) -> None:
        """Load the latest model and optimizer from the last_training_config.pt file if it exists, otherwise start from scratch."""
        try:
            with open(self.save_path / 'last_training_config.json', 'r') as f:
                last_training_config = json.load(f)

            new_starting_iteration = int(last_training_config['iteration'])
            if self.starting_iteration == new_starting_iteration:
                print(f'No new model found, starting from iteration {self.starting_iteration}')
                return

            self.starting_iteration = new_starting_iteration
            self.model.load_state_dict(
                torch.load(last_training_config['model'], map_location=self.model.device, weights_only=True)
            )
            self.optimizer.load_state_dict(
                torch.load(last_training_config['optimizer'], map_location=self.model.device, weights_only=True)
            )

            clear_cache()

            print(f'Model and optimizer loaded from iteration {self.starting_iteration}')
        except FileNotFoundError:
            print('No model and optimizer found, starting from scratch')

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

        print(f'Model and optimizer saved at iteration {iteration}')

    def _save_memory(self, memory: list[SelfPlayMemory], iteration: int) -> None:
        memory_path = self.save_path / f'memory_{iteration}_{random_id()}.pt'
        torch.save(
            [(mem.state, mem.policy_targets, mem.value_targets) for mem in memory],
            memory_path,
        )
        print(f'Memory saved at iteration {iteration}')

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
