from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset
from torch.multiprocessing import Process

from src.Encoding import decode_board_state
from src.mcts.MCTS import action_probabilities
from src.settings import log_histogram, log_scalar
from src.util.log import log
from src.util.tensorboard import TensorboardWriter
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats

# TODO: The entire selfplaytraindataset should fit into Memory at once, with compressed state and Action probs. This could speed up training by reducing the number of file reads and writes


class SelfPlayTrainDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset to train the neural network on self-play data. It is a wrapper around multiple SelfPlayDatasets (i.e. Iterations). The Idea is, to load only chunks of the datasets into memory and return the next sample from the next dataset in a round-robin fashion."""

    def __init__(self, run: int) -> None:
        self.run = run

        self.all_chunks: list[list[Path]] = []

        self.stats = SelfPlayDatasetStats()

    def load_from_files(
        self,
        folder_path: str,
        origins: list[tuple[int, list[Path]]],
        max_num_repetitions: int,
        fraction_of_samples: float = 1.0,
    ) -> None:
        self.stats = SelfPlayDatasetStats()

        total_samples_pre_deduplication = 0
        total_samples_post_deduplication = 0

        for i in range(max_num_repetitions):
            for iteration, sublist in origins[::-1]:
                if len(sublist) == 0:
                    continue

                iteration_dataset = SelfPlayDataset()
                for file in sublist:
                    processed_indicator = file.parent / (file.stem + '.processed')
                    if processed_indicator.exists():
                        continue
                    iteration_dataset += SelfPlayDataset.load(file)
                    processed_indicator.touch()

                total_samples_pre_deduplication += len(iteration_dataset)
                iteration_dataset = iteration_dataset.deduplicate()
                total_samples_post_deduplication += len(iteration_dataset)

                iteration_dataset = iteration_dataset.sample(int(len(iteration_dataset) * fraction_of_samples))
                iteration_dataset = iteration_dataset.shuffle()
                iteration_dataset.chunked_save(folder_path + '/shuffled', iteration, 500)

                chunks = SelfPlayDataset.get_files_to_load_for_iteration(folder_path + '/shuffled', iteration)
                if not chunks:
                    log(f'No chunks found for iteration {iteration}.')
                    continue

                for chunk in chunks:
                    self.stats += SelfPlayDataset.load_stats(chunk)

                self.all_chunks.append(chunks)

            if self.stats.num_samples > 5_000_000:
                log(f'Loaded {self.stats.num_samples} samples with {i+1} multiplications.')
                break

        thread = Process(
            target=self._log_all_dataset_stats,
            args=([list.copy() for list in self.all_chunks], self.run),
            daemon=True,
        )
        thread.start()
        log(
            f'Initially loaded {total_samples_pre_deduplication} samples. After deduplication {total_samples_post_deduplication} samples remained.'
        )

    @staticmethod
    def _log_all_dataset_stats(iterations: list[list[Path]], run: int) -> None:
        accumulated_stats = SelfPlayDatasetStats()

        with TensorboardWriter(run, 'dataset', postfix_pid=False):
            for files in iterations:
                for file in files:
                    dataset = SelfPlayDataset.load(file)
                    if len(dataset) == 0:
                        continue

                    accumulated_stats += dataset.stats

                    policies = [action_probabilities(visits) for visits in dataset.visit_counts]

                    spikiness = sum(policy.max() for policy in policies) / len(dataset)

                    log_scalar('dataset/policy_spikiness', spikiness)

                    log_histogram('dataset/policy_targets', np.array([policy.max() for policy in policies]))
                    log_histogram('dataset/value_targets', np.array(dataset.value_targets))

                    # dataset.deduplicate()
                    # log_histogram('dataset/value_targets_deduplicated', np.array(dataset.value_targets))

            log_scalar('dataset/num_games', accumulated_stats.num_games)
            log_scalar('dataset/average_game_length', accumulated_stats.game_lengths / accumulated_stats.num_games)
            log_scalar('dataset/num_too_long_games', accumulated_stats.num_too_long_games)
            log_scalar('dataset/num_resignations', accumulated_stats.resignations)
            log_scalar(
                'dataset/average_resignation_percent',
                accumulated_stats.resignations / accumulated_stats.num_games * 100,
            )
            log_scalar('dataset/num_samples', accumulated_stats.num_samples)
            log_scalar(
                'dataset/average_generation_time', accumulated_stats.total_generation_time / accumulated_stats.num_games
            )

    def as_dataloader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
        assert num_workers > 0, 'num_workers must be greater than 0'
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=8,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            assert False, 'For now, this dataset is only compatible with DataLoader with num_workers > 0'

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        chunks = self.all_chunks[worker_id::num_workers]
        self.sample_index = [0] * len(chunks)

        active_chunks = [SelfPlayDataset.load(chunk[0]) for chunk in chunks]

        while chunks:
            for i, chunk in enumerate(chunks):
                while chunk and self.sample_index[i] >= len(active_chunks[i]):
                    active_chunks[i] = SelfPlayDataset.load(chunk.pop(0))
                    self.sample_index[i] = 0

                if not chunk or self.sample_index[i] >= len(active_chunks[i]):
                    continue

                dataset = active_chunks[i]
                state = torch.from_numpy(decode_board_state(dataset.encoded_states[self.sample_index[i]]))
                policy_target = torch.from_numpy(action_probabilities(dataset.visit_counts[self.sample_index[i]]))
                value_target = torch.tensor(dataset.value_targets[self.sample_index[i]])
                self.sample_index[i] += 1

                yield state, policy_target, value_target

            chunks = [chunk for chunk in chunks if chunk]

    def __len__(self) -> int:
        return self.stats.num_samples
