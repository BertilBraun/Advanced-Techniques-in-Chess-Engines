import json
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm, trange
from pathlib import Path

from AIZeroConnect4Bot.src.Network import NN_CACHE, TOTAL_EVALS, TOTAL_HITS, Network
from AIZeroConnect4Bot.src.SelfPlay import SelfPlay, SelfPlayMemory
from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.TrainingStats import TrainingStats
from AIZeroConnect4Bot.src.util import batched_iterate, random_id
from AIZeroConnect4Bot.src.hashing import zobrist_hash_boards


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
            train_stats += self._train(memory, iteration)
            print(f'Epoch {epoch + 1}: {train_stats}')

        print(f'Iteration {iteration + 1}: {train_stats}')
        self._save_latest_model(iteration)
        return train_stats

    def _train(self, memory: list[SelfPlayMemory], iteration: int) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """
        random.shuffle(memory)

        train_stats = TrainingStats()
        base_lr = self.args.learning_rate(iteration)

        self.model.train()

        for batchIdx, sample in tqdm(
            enumerate(batched_iterate(memory, self.args.batch_size)),
            desc='Training batches',
            total=len(memory) // self.args.batch_size,
        ):
            state = [[mem.state] for mem in sample]  # add a dimension to the state to allow the conv layers to work
            policy_targets = [mem.policy_targets for mem in sample]
            value_targets = [mem.value_targets for mem in sample]

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Update learning rate before stepping the optimizer
            lr = self.args.learning_rate_scheduler(batchIdx / len(memory), base_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        return train_stats

    def _deduplicate_positions(self, memory: list[SelfPlayMemory]) -> list[SelfPlayMemory]:
        """Deduplicate the positions in the memory by averaging the policy and value targets for the same board state."""
        mp: dict[int, tuple[int, SelfPlayMemory]] = {}
        for batch in batched_iterate(memory, 128):
            states = [mem.state for mem in batch]
            hashes = zobrist_hash_boards(torch.stack(states))

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

            print('Cache hit rate:', TOTAL_HITS / TOTAL_EVALS, 'on cache size', len(NN_CACHE))
            NN_CACHE.clear()

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
        torch.save(memory, memory_path)
        print(f'Memory saved at iteration {iteration}')

    def _load_all_memories_to_train_on_for_iteration(self, iteration: int) -> list[SelfPlayMemory]:
        window_size = self.args.sampling_window(iteration)

        memory: list[SelfPlayMemory] = []
        for iter in range(max(iteration - window_size, 0), iteration):
            memory += self._load_all_memories(iter)
        return memory

    def _load_all_memories(self, iteration: int) -> list[SelfPlayMemory]:
        memory: list[SelfPlayMemory] = []
        for f in self.save_path.iterdir():
            if f.suffix == '.pt' and f.stem.startswith(f'memory_{iteration}_'):
                memory += torch.load(f)
        return memory
