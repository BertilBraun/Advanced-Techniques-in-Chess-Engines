# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

from pathlib import Path
import random
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from AIZeroChessBot.src.Network import Network
from AIZeroChessBot.src.SelfPlay import SelfPlay, SelfPlayMemory
from AIZeroChessBot.src.TrainingArgs import TrainingArgs
from AIZeroChessBot.src.TrainingStats import LearningStats, TrainingStats


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

        if load_latest_model:
            self._load_latest_model()

    def learn(self) -> None:
        old_memory: list[SelfPlayMemory] = []
        learning_stats = LearningStats()

        for iteration in range(self.starting_iteration, self.args.num_iterations):
            train_stats = TrainingStats()
            memory: list[SelfPlayMemory] = []

            self.model.eval()
            for _ in tqdm(range(self.args.num_self_play_iterations // self.args.num_parallel_games), desc='Self Play'):
                memory += self.self_play.self_play()

            total_games = len(memory)
            print(f'Collected {total_games} self-play memories')
            # TODO save memories to disk with the iteration number

            self.model.train()
            for _ in tqdm(range(self.args.num_epochs), desc='Training'):
                train_stats += self._train(memory + old_memory)

            print(f'Iteration {iteration + 1}: {train_stats}')
            self._save(iteration)
            learning_stats.update(total_games, train_stats)

            # drop 75% of the memory
            # retain 25% of the memory for the next iteration to train on and not overfit to the current iteration
            old_memory = random.sample(memory, int(total_games * 0.25))

        print(learning_stats)

    def _train(self, memory: list[SelfPlayMemory]) -> TrainingStats:
        """
        Train the model with the given memory.
        The target is the policy and value targets from the self-play memory.
        The model is trained to minimize the cross-entropy loss for the policy and the mean squared error for the value when evaluated on the board state from the memory.
        """

        random.shuffle(memory)

        train_stats = TrainingStats()

        for batchIdx in range(0, len(memory), self.args.batch_size):
            sample = memory[batchIdx : min(batchIdx + self.args.batch_size, len(memory) - 1)]

            state = [mem.state for mem in sample]
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
            last_training_config = torch.load('last_training_config.pt')
            self.model.load_state_dict(torch.load(last_training_config['model']))
            self.optimizer.load_state_dict(torch.load(last_training_config['optimizer']))
            self.starting_iteration = last_training_config['iteration']
            print(f'Model and optimizer loaded from iteration {self.starting_iteration}')
        except FileNotFoundError:
            print('No model and optimizer found, starting from scratch')

    def _save(self, iteration: int) -> None:
        """Save the model and optimizer to the current directory with the current iteration number. Also save the current training configuration to last_training_config.pt."""

        save_dir = Path(self.args.save_path)
        model_path = save_dir / f'model_{iteration}.pt'
        optimizer_path = save_dir / f'optimizer_{iteration}.pt'
        last_training_config_path = save_dir / 'last_training_config.pt'

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        torch.save(
            {
                'model': f'model_{iteration}.pt',
                'optimizer': f'optimizer_{iteration}.pt',
                'iteration': iteration,
            },
            last_training_config_path,
        )
        print(f'Model and optimizer saved at iteration {iteration}')
