from __future__ import annotations

import torch
import asyncio
import numpy as np
from sys import getsizeof
from typing import Any, Coroutine, TypeVar

from src.Encoding import encode_board_state
from src.Network import Network
from src.train.TrainingArgs import TrainingArgs
from src.settings import TORCH_DTYPE, USE_GPU, CurrentBoard, CurrentGame, log_histogram, log_scalar
from src.util.log import log
from src.util.timing import timeit
from src.util.save_paths import load_model, model_save_path


T = TypeVar('T')


class InferenceClient:
    """The Inference Client is responsible for batching and caching inference requests. It uses a model to directly infer the policy and value for a given board state on the provided device."""

    def __init__(self, device_id: int, args: TrainingArgs) -> None:
        self.args = args
        self.model: Network = None  # type: ignore
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')

        self.batch_queue: list[tuple[np.ndarray | None, bytes, asyncio.Future]] = []

        self.inference_cache: dict[bytes, tuple[np.ndarray, float]] = {}
        self.total_hits = 0
        self.total_evals = 0

        self.enqueued_inferences: set[bytes] = set()

    def update_iteration(self, iteration: int) -> None:
        """Update the Inference Client to use the model for the given iteration.
        Slighly optimizes the model for inference and resets the cache and stats."""

        self.model = load_model(model_save_path(iteration, self.args.save_path), self.args.network, self.device)
        # self.model = load_model(R'C:\Users\berti\OneDrive\Desktop\zip6\zip\model_64.pt', self.args.network, self.device)
        self.model.disable_auto_grad()
        self.model = self.model.eval()
        self.model.fuse_model()

        if self.total_evals != 0:
            cache_hit_rate = (self.total_hits / self.total_evals) * 100
            log_scalar('cache/hit_rate', cache_hit_rate, iteration)
            log_scalar('cache/unique_positions', len(self.inference_cache), iteration)
            log_histogram(
                'nn_output_value_distribution',
                np.array([round(v, 1) for _, v in self.inference_cache.values()]),
                iteration,
            )

            size_in_mb = 0
            for key, (policy, value) in self.inference_cache.items():
                size_in_mb += getsizeof(key) + getsizeof(value) + policy.nbytes

            size_in_mb /= 1024 * 1024
            log_scalar('cache/size_mb', size_in_mb, iteration)
            log(
                f'Cache hit rate: {cache_hit_rate:.2f}% on cache size {len(self.inference_cache)} ({size_in_mb:.2f} MB)'
            )
        self.inference_cache.clear()

    @timeit
    async def inference(self, board: CurrentBoard) -> tuple[np.ndarray, float]:
        """Enqueue an inference request and return the result when available.
        Results are batched to optimize the inference process.
        Results are cached to avoid redundant inferences.
        Results will be available once either the batch size is reached or the flush method is called."""

        canonical_board = CurrentGame.get_canonical_board(board)
        board_hash = _get_board_hash(canonical_board)

        self.total_evals += 1

        if board_hash in self.inference_cache:
            self.total_hits += 1
            return self.inference_cache[board_hash]

        # Create a Future to wait for the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Enqueue the request
        if board_hash not in self.enqueued_inferences:
            self.enqueued_inferences.add(board_hash)
            self.batch_queue.append((canonical_board, board_hash, future))
        else:
            self.total_hits += 1
            self.batch_queue.append((None, board_hash, future))

        # If the batch size is reached, process the batch immediately
        current_batch_size = len(self.enqueued_inferences)
        if current_batch_size >= self.args.inference.batch_size:
            # Make a copy of the current batch to avoid race conditions
            self._process_batch()

        # Await the result
        return await future

    async def run_batch(self, tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
        """Run a batch of inference requests and return the results. The order of the results is preserved."""
        requests = [asyncio.create_task(task) for task in tasks]
        await asyncio.sleep(0)  # Yield control to allow tasks to enqueue
        self.flush()
        return await asyncio.gather(*requests)

    def flush(self) -> None:
        """Flush the batch queue and process all enqueued inferences."""
        if self.batch_queue:
            self._process_batch()

    @timeit
    def _process_batch(self) -> None:
        encoded_boards = [board for board, _, _ in self.batch_queue if board is not None]
        board_hashes = [hash for board, hash, _ in self.batch_queue if board is not None]

        if encoded_boards:
            results = self._model_inference(encoded_boards)

            for hash, (policy, value) in zip(board_hashes, results):
                self.inference_cache[hash] = policy, value

        for _, hash, future in self.batch_queue:
            assert not future.cancelled() and not future.done(), 'Future is already done'
            future.set_result(self.inference_cache[hash])

        self.batch_queue.clear()
        self.enqueued_inferences.clear()

    @timeit
    @torch.no_grad()
    def _model_inference(self, boards: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        input_tensor = torch.tensor(np.array(boards), dtype=TORCH_DTYPE, device=self.device)

        policies, values = self.model(input_tensor)

        policies = torch.softmax(policies, dim=1)
        values = torch.mean(values, dim=1)

        results = torch.cat((policies, values.unsqueeze(1)), dim=1)

        results = results.to(dtype=torch.float32, device='cpu').numpy()

        # return [
        #     (np.full_like(result[:-1], 1 / CurrentGame.action_size), 0.0)  # TODO
        #     for result in results
        # ]
        return [(result[:-1], result[-1]) for result in results]


@timeit
def _get_board_hash(board: np.ndarray) -> bytes:
    variation_hashes: list[bytes] = []
    # TODO assuming, that a vertical flip is a valid symmetry
    for variation in (board, np.flip(board, axis=2)):
        variation_hashes.append(encode_board_state(variation).tobytes())
    return min(variation_hashes)
