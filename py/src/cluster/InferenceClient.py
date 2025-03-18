from __future__ import annotations
from os import PathLike
from random import random
from time import sleep

import torch
import numpy as np
from sys import getsizeof
from typing import TypeVar

from src.Encoding import MoveScore, filter_policy_with_en_passant_moves_then_get_moves_and_probabilities
from src.Network import Network
from src.train.TrainingArgs import NetworkParams
from src.settings import TORCH_DTYPE, USE_GPU, CurrentBoard, CurrentGame, log_histogram, log_scalar
from src.util.ZobristHasherNumpy import ZobristHasherNumpy
from src.util.log import LogLevel, log
from src.util.timing import timeit
from src.util.save_paths import load_model, model_save_path


T = TypeVar('T')


MoveList = np.ndarray  #  list[MoveScore]


class InferenceClient:
    """The Inference Client is responsible for batching and caching inference requests. It uses a model to directly infer the policy and value for a given board state on the provided device."""

    def __init__(self, device_id: int, network_args: NetworkParams, save_path: str) -> None:
        self.network_args = network_args
        self.save_path = save_path
        self.model: Network = None  # type: ignore
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')

        self.inference_cache: dict[int, tuple[MoveList, float]] = {}
        self.total_hits = 0
        self.total_evals = 0

        channels, rows, cols = CurrentGame.representation_shape
        self.hasher = ZobristHasherNumpy(channels, rows, cols)

    def load_model(self, model_path: str | PathLike) -> None:
        for _ in range(5):
            try:
                if hasattr(self, 'model'):
                    del self.model

                # sync and gc collect to free up memory before loading the model
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                self.model = load_model(model_path, self.network_args, self.device)
                self.model.disable_auto_grad()
                self.model.eval()
                self.model.fuse_model()
                break
            except RuntimeError:
                # log(f'Failed to load model: "{e}" retrying...', level=LogLevel.ERROR)
                sleep(random() * 60)  # sleep for a random amount of time to avoid overloading the GPU VRAM
        else:
            raise RuntimeError('Failed to load model after 5 retries')

    def update_iteration(self, iteration: int) -> None:
        """Update the Inference Client to use the model for the given iteration.
        Slighly optimizes the model for inference and resets the cache and stats."""

        self.load_model(model_save_path(iteration, self.save_path))

        if self.total_evals != 0:
            cache_hit_rate = (self.total_hits / self.total_evals) * 100
            log_scalar('cache/hit_rate', cache_hit_rate, iteration)
            log_scalar('cache/unique_positions', len(self.inference_cache), iteration)
            log_histogram(
                'nn_output_value_distribution',
                np.array([v for _, v in self.inference_cache.values()]),
                iteration,
            )

            size_in_mb = 0
            for key, (policy, value) in self.inference_cache.items():
                size_in_mb += getsizeof(key) + getsizeof(value) + getsizeof(policy) + getsizeof(policy[0]) * len(policy)

            size_in_mb /= 1024 * 1024
            log_scalar('cache/size_mb', size_in_mb, iteration)
            log(
                f'Cache hit rate: {cache_hit_rate:.2f}% on cache size {len(self.inference_cache)} ({size_in_mb:.2f} MB)',
                level=LogLevel.DEBUG,
            )
        self.inference_cache.clear()

    def inference_batch(self, boards: list[CurrentBoard]) -> list[tuple[list[MoveScore], float]]:
        if not boards:
            return []

        encoded_boards = [CurrentGame.get_canonical_board(board) for board in boards]
        board_hashes = self.hasher.zobrist_hash_boards(np.array(encoded_boards))

        boards_to_infer: list[np.ndarray] = []
        inference_hashes_and_boards: list[tuple[int, CurrentBoard]] = []

        enqueued_hashes: set[int] = set()  # for O(1) lookup of enqueued hashes

        for board, encoded_board, hash in zip(boards, encoded_boards, board_hashes):
            if hash not in enqueued_hashes and hash not in self.inference_cache:
                enqueued_hashes.add(hash)
                boards_to_infer.append(encoded_board)
                inference_hashes_and_boards.append((hash, board))
            else:
                self.total_hits += 1

        self.total_evals += len(boards)

        if boards_to_infer:
            results = self._model_inference(boards_to_infer)
            for (hash, board), (policy, value) in zip(inference_hashes_and_boards, results):
                # Dont filter out the en passant moves here, because the policy is already fixed, but the en passant moves are dependent on the board state
                moves = filter_policy_with_en_passant_moves_then_get_moves_and_probabilities(policy, board)
                self.inference_cache[hash] = np.array(moves), value

        responses: list[tuple[list[MoveScore], float]] = []
        for hash in board_hashes:
            moves_np, value = self.inference_cache[hash]
            moves = [(int(move), float(prob)) for move, prob in moves_np]
            responses.append((moves, value))

        return responses

    @timeit
    @torch.no_grad()
    def _model_inference(self, boards: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        if self.model is None:
            if not hasattr(self, 'WARNING_SHOWN'):
                self.WARNING_SHOWN = True
                log('Model not loaded', level=LogLevel.ERROR)
            return [(np.full((CurrentGame.action_size,), 1 / CurrentGame.action_size), 0.0) for _ in boards]

        input_tensor = torch.tensor(np.array(boards), dtype=TORCH_DTYPE, device=self.device)

        policies, values = self.model(input_tensor)

        policies = torch.softmax(policies, dim=1)

        results = torch.cat((policies, values), dim=1)

        results = results.to(dtype=torch.float32, device='cpu').numpy()

        return [(result[:-1], result[-1]) for result in results]
