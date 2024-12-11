# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

import json
import time
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from AIZeroConnect4Bot.src.Network import Network
from AIZeroConnect4Bot.src.SelfPlay import SelfPlay, SelfPlayMemory
from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs
from AIZeroConnect4Bot.src.TrainingStats import LearningStats, TrainingStats


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

        self._initialize_cluster()

    def learn(self) -> None:
        old_memory: list[SelfPlayMemory] = []
        learning_stats = LearningStats()

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            train_stats = TrainingStats()
            memory: list[SelfPlayMemory] = []

            self_play_games_in_parallel = self.args.num_parallel_games * self.args.num_separate_nodes_on_cluster
            self_play_iterations = self.args.num_self_play_iterations // (self_play_games_in_parallel)

            self.model.eval()
            for _ in tqdm(
                range(self_play_iterations),
                desc=f'Self Play for {self_play_games_in_parallel} games in parallel',
            ):
                memory += self.self_play.self_play()

            total_games = len(memory)
            print(f'Collected {total_games} self-play memories')
            self._save_memory(memory, iteration)

            if self.is_root_node:
                memory = self._load_all_memories(iteration)

                print(f'Training with {len(memory)} self-play memories')

                self.model.train()
                for epoch in range(self.args.num_epochs):
                    train_stats += self._train(memory + old_memory)
                    print(f'Epoch {epoch + 1}: {train_stats}')

                print(f'Iteration {iteration + 1}: {train_stats}')
                model_path, _, _ = self._save_latest_model(iteration)
                learning_stats.update(total_games, train_stats)

                # drop 75% of the memory
                # retain 25% of the memory for the next iteration to train on and not overfit to the current iteration
                old_memory = random.sample(memory, int(total_games * 0.25))

                # evaluate_alpha_vs_stockfish(model_path)

            self.barrier('training_done')

            self._load_latest_model()

        print(learning_stats)

    def _train(self, memory: list[SelfPlayMemory]) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """

        random.shuffle(memory)

        train_stats = TrainingStats()

        for batchIdx in tqdm(range(0, len(memory), self.args.batch_size), desc='Training batches'):
            sample = memory[batchIdx : min(batchIdx + self.args.batch_size, len(memory) - 1)]

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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_stats.update(policy_loss.item(), value_loss.item(), loss.item())

        return train_stats

    def _load_latest_model(self) -> None:
        """Load the latest model and optimizer from the last_training_config.pt file if it exists, otherwise start from scratch."""
        try:
            with open(self.save_path / 'last_training_config.json', 'r') as f:
                last_training_config = json.load(f)

            self.model.load_state_dict(
                torch.load(last_training_config['model'], map_location=self.model.device, weights_only=True)
            )
            self.optimizer.load_state_dict(
                torch.load(last_training_config['optimizer'], map_location=self.model.device, weights_only=True)
            )
            self.starting_iteration = int(last_training_config['iteration'])

            print(f'Model and optimizer loaded from iteration {self.starting_iteration}')
        except FileNotFoundError:
            print('No model and optimizer found, starting from scratch')

    def _save_latest_model(self, iteration: int) -> tuple[Path, Path, Path]:
        """Save the model and optimizer to the current directory with the current iteration number. Also save the current training configuration to last_training_config.pt."""

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

        return model_path, optimizer_path, last_training_config_path

    def _save_memory(self, memory: list[SelfPlayMemory], iteration: int) -> None:
        memory_path = self.save_path / f'memory_{iteration}_{self.my_id}.pt'
        torch.save(memory, memory_path)
        print(f'Memory saved at iteration {iteration}')

        self.barrier('memory_saved')

    def _load_all_memories(self, iteration: int) -> list[SelfPlayMemory]:
        memory = []
        for f in self.save_path.iterdir():
            if f.suffix == '.pt' and f.stem.startswith(f'memory_{iteration}_'):
                memory += torch.load(f)
        return memory

    def barrier(self, name: str) -> None:
        log_file = self.communication_dir / f'{self.my_id}{"_" + name if name else ""}.txt'
        open(log_file, 'w').close()

        print(f'Node {self.my_id} reached the barrier {name}')
        while True:
            written_files = len([f for f in self.communication_dir.iterdir() if f.stem.endswith(name)])

            time.sleep(3)

            if written_files == self.args.num_separate_nodes_on_cluster:
                break

        # remove the memory saved file
        log_file.unlink(missing_ok=True)

        print(f'All nodes have reached the barrier {name}')

    def _initialize_cluster(self) -> None:
        """Initialize the cluster for parallel self-play."""
        self.my_id = random.randint(0, 1000000000)

        self.communication_dir = Path('communication')
        self.communication_dir.mkdir(exist_ok=True)
        # delete everything in the communication directory
        for f in self.communication_dir.iterdir():
            f.unlink(missing_ok=True)

        print(f'Node {self.my_id} initialized')

        log_file = self.communication_dir / f'{self.my_id}.txt'

        while True:
            open(log_file, 'w').close()

            initialized_nodes = len(list(self.communication_dir.iterdir()))
            if initialized_nodes == self.args.num_separate_nodes_on_cluster:
                break

            print(f'Waiting for {self.args.num_separate_nodes_on_cluster - initialized_nodes} nodes to initialize')
            time.sleep(1)

        print('All nodes initialized')

        # root is the node with the lowest id
        if self.my_id == min(int(f.stem) for f in self.communication_dir.iterdir()):
            print('I am the root node')
            self.is_root_node = True
        else:
            print('I am not the root node')
            self.is_root_node = False

        # TODO time.sleep(10)

        # remove the initialization file
        log_file.unlink(missing_ok=True)
