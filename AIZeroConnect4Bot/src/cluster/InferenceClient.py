from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar
import numpy as np
import torch

from src.Encoding import encode_board_state
from src.Network import Network
from src.alpha_zero.train.TrainingArgs import InferenceParams, NetworkParams
from src.settings import TORCH_DTYPE, USE_GPU, CurrentBoard, CurrentGame
from src.util.save_paths import load_model, model_save_path


T = TypeVar('T')


class InferenceClient:
    def __init__(self, device_id: int, network_args: NetworkParams, inference_args: InferenceParams) -> None:
        self.model: Network = None  # type: ignore
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')

        self.network_args = network_args
        self.inference_args = inference_args

        self.batch_queue: list[tuple[np.ndarray, bytes, asyncio.Future]] = []

        self.inference_cache: dict[bytes, tuple[np.ndarray, float]] = {}

    def update_iteration(self, iteration: int) -> None:
        self.model = load_model(model_save_path(iteration), self.network_args, self.device)
        self.model = self.model.eval()

        self.inference_cache.clear()

    async def inference(self, board: CurrentBoard) -> tuple[np.ndarray, float]:
        canonical_board = CurrentGame.get_canonical_board(board)
        hash = encode_board_state(canonical_board).tobytes()

        if hash in self.inference_cache:
            return self.inference_cache[hash]

        # Create a Future to wait for the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Enqueue the request
        self.batch_queue.append((canonical_board, hash, future))
        current_batch_size = len(self.batch_queue)

        # If the batch size is reached, process the batch immediately
        if current_batch_size >= self.inference_args.batch_size:
            # Make a copy of the current batch to avoid race conditions
            batch_to_process = self.batch_queue[: self.inference_args.batch_size]
            self.batch_queue = self.batch_queue[self.inference_args.batch_size :]
            # Schedule batch processing without waiting
            self._process_batch(batch_to_process)

        # Await the result
        return await future

    def flush(self) -> None:
        if self.batch_queue:
            self._process_batch(self.batch_queue)
            self.batch_queue.clear()

    @torch.no_grad()
    def _process_batch(self, batch: list[tuple[np.ndarray, bytes, asyncio.Future]]) -> None:
        encoded_boards = [board for board, _, _ in batch]
        input_tensor = torch.tensor(np.array(encoded_boards), dtype=TORCH_DTYPE, device=self.device)

        policies, values = self.model(input_tensor)

        policies = torch.softmax(policies, dim=1)
        values = torch.mean(values, dim=1)

        results = torch.cat((policies, values.unsqueeze(1)), dim=1)

        results = results.to(dtype=torch.float32, device='cpu').numpy()

        for (_, hash, future), result in zip(batch, results):
            policy, value = result[:-1], result[-1]
            self.inference_cache[hash] = policy, value
            assert not future.cancelled() and not future.done(), 'Future is already done'
            future.set_result((policy, value))

    async def run_batch(self, tasks: list[Coroutine[Any, Any, T]]) -> list[T]:
        requests = [asyncio.create_task(task) for task in tasks]
        await asyncio.sleep(0)  # Yield control to allow tasks to enqueue
        self.flush()
        return await asyncio.gather(*requests)
