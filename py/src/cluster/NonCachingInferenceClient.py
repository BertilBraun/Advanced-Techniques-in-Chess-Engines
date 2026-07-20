from __future__ import annotations

import gc
from os import PathLike
from random import random
from time import sleep

import numpy as np
import torch

from src.Encoding import MoveScore, filter_policy_then_get_moves_and_probabilities
from src.Network import Network
from src.settings import USE_GPU, CurrentBoard, CurrentGame
from src.train.TrainingArgs import NetworkParams
from src.util.log import error, warn
from src.util.save_paths import load_model, model_save_path
from src.util.timing import timeit
from src.value import wdl_to_scalar


class NonCachingInferenceClient:
    """Runs each requested board directly through the model without cache bookkeeping."""

    def __init__(self, device_id: int, network_args: NetworkParams, save_path: str) -> None:
        self.network_args = network_args
        self.save_path = save_path
        self.model: Network | None = None
        self.device = torch.device('cuda', device_id) if USE_GPU else torch.device('cpu')
        self.dtype = torch.bfloat16 if USE_GPU else torch.float32
        self.warning_shown = False

    def load_model(self, model_path: str | PathLike[str]) -> None:
        for _ in range(5):
            try:
                self.model = None
                if USE_GPU:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                model = load_model(model_path, self.network_args, self.device)
                model.to(dtype=self.dtype, device=self.device, non_blocking=True)
                model.disable_auto_grad()
                model.eval()
                model.fuse_model()
                self.model = model
                return
            except RuntimeError as exception:
                warn(f'Failed to load model: "{exception}" retrying...')
                sleep(random() * 60)
        raise RuntimeError('Failed to load model after 5 retries')

    def update_iteration(self, iteration: int) -> None:
        self.load_model(model_save_path(iteration, self.save_path))

    def inference_batch(self, boards: list[CurrentBoard]) -> list[tuple[list[MoveScore], float]]:
        if not boards:
            return []

        encoded_boards = [CurrentGame.get_canonical_board(board) for board in boards]
        results = self._model_inference(encoded_boards)
        return [
            (filter_policy_then_get_moves_and_probabilities(policy, board), value)
            for board, (policy, value) in zip(boards, results)
        ]

    @timeit
    @torch.no_grad()
    def _model_inference(self, boards: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        if self.model is None:
            if not self.warning_shown:
                self.warning_shown = True
                error('Model not loaded')
            return [(np.full((CurrentGame.action_size,), 1 / CurrentGame.action_size), 0.0) for _ in boards]

        input_tensor = torch.from_numpy(np.array(boards)).to(
            dtype=self.dtype,
            device=self.device,
            non_blocking=True,
        )
        policies, value_probabilities = self.model(input_tensor)
        policies = torch.detach(policies).to(dtype=torch.float32, device='cpu').numpy()
        values = torch.detach(wdl_to_scalar(value_probabilities)).to(dtype=torch.float32, device='cpu').numpy()
        return [(policy, float(value)) for policy, value in zip(policies, values)]
