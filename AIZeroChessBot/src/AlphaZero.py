# This bot is heavily based on the Alpha Zero From Scratch Project by foersterrober (https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/9.AlphaParallel.ipynb)

from pathlib import Path
import random
import time
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from AIZeroChessBot.eval.EvaluateAlphaVsStockfish import evaluate_alpha_vs_stockfish

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

                self.model.train()
                for _ in tqdm(range(self.args.num_epochs), desc='Training'):
                    train_stats += self._train(memory + old_memory)

                print(f'Iteration {iteration + 1}: {train_stats}')
                model_path, _, _ = self._save_latest_model(iteration)
                learning_stats.update(total_games, train_stats)

            self.barrier('training_done')

            self._load_latest_model()

            # drop 75% of the memory
            # retain 25% of the memory for the next iteration to train on and not overfit to the current iteration
            old_memory = random.sample(memory, int(total_games * 0.25))

            evaluate_alpha_vs_stockfish(model_path)

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

    def _save_latest_model(self, iteration: int) -> tuple[Path, Path, Path]:
        """Save the model and optimizer to the current directory with the current iteration number. Also save the current training configuration to last_training_config.pt."""

        save_dir = Path(self.args.save_path)
        model_path = save_dir / f'model_{iteration}.pt'
        optimizer_path = save_dir / f'optimizer_{iteration}.pt'
        last_training_config_path = save_dir / 'last_training_config.pt'

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        torch.save(
            {
                'model': model_path,
                'optimizer': optimizer_path,
                'iteration': iteration,
            },
            last_training_config_path,
        )
        print(f'Model and optimizer saved at iteration {iteration}')

        return model_path, optimizer_path, last_training_config_path

    def _save_memory(self, memory: list[SelfPlayMemory], iteration: int) -> None:
        save_dir = Path(self.args.save_path)
        memory_path = save_dir / f'memory_{iteration}_{self.my_id}.pt'
        torch.save(memory, memory_path)
        print(f'Memory saved at iteration {iteration}')

        self.barrier('memory_saved')

    def _load_all_memories(self, iteration: int) -> list[SelfPlayMemory]:
        memory = []
        for f in Path(self.args.save_path).iterdir():
            if f.suffix == '.pt' and f.stem.startswith(f'memory_{iteration}_'):
                memory += torch.load(f)
        return memory

    def barrier(self, name: str = '') -> None:
        open(self.communication_dir / f'{self.my_id}{"_" + name if name else ""}.txt', 'w').close()

        while True:
            written_files = len(list(self.communication_dir.iterdir()))
            if written_files == self.args.num_separate_nodes_on_cluster:
                break

            print(f'Waiting for {self.args.num_separate_nodes_on_cluster - written_files} nodes')
            time.sleep(5)

        time.sleep(10)

        # remove the memory saved files
        for f in self.communication_dir.iterdir():
            f.unlink()

        print('All nodes have reached the barrier')

    def _initialize_cluster(self) -> None:
        """Initialize the cluster for parallel self-play."""
        self.my_id = random.randint(0, 1000000000)

        self.communication_dir = Path('communication')
        # delete the communication directory if it exists
        if self.communication_dir.exists():
            for f in self.communication_dir.iterdir():
                f.unlink()
            self.communication_dir.rmdir()

        print(f'Node {self.my_id} initialized')

        while True:
            self.communication_dir.mkdir(exist_ok=True)
            try:
                open(self.communication_dir / f'{self.my_id}.txt', 'w').close()
            except:  # noqa: E722
                print(f'Node {self.my_id} failed to write to the communication directory')
                pass

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

        time.sleep(5)

        # remove the initialization files
        for f in self.communication_dir.iterdir():
            f.unlink()
