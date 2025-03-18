from __future__ import annotations
from collections import defaultdict
import contextlib
import zmq
import time
import pickle


from os import PathLike

import torch
import numpy as np
from typing import TypeVar

from src.Encoding import decode_board_state, filter_policy_with_en_passant_moves_then_get_moves_and_probabilities
from src.Network import Network
from src.cluster.InferenceCache import InferenceCache
from src.train.TrainingArgs import NetworkParams
from src.settings import TORCH_DTYPE, TRAINING_ARGS, USE_GPU, CurrentGame
from src.util.ZobristHasherNumpy import ZobristHasherNumpy
from src.util.exceptions import log_exceptions
from src.util.log import LogLevel, log
from src.util.tensorboard import TensorboardWriter
from src.util.timing import timeit
from src.util.save_paths import get_latest_model_iteration, load_model, model_save_path


# Settings for batching
BATCH_SIZE = 256  # Adjust up to 1024 if needed.
BATCH_TIMEOUT = 0.03  # 30ms timeout

T = TypeVar('T')


MoveList = np.ndarray  #  list[MoveScore]


class BasicModelInference:
    """The Inference Client is responsible for batching and caching inference requests. It uses a model to directly infer the policy and value for a given board state on the provided device."""

    def __init__(self, device_id: int, network_args: NetworkParams, save_path: str) -> None:
        self.network_args = network_args
        self.save_path = save_path
        self.model: Network = None  # type: ignore
        self.device = torch.device('cuda', device_id % torch.cuda.device_count()) if USE_GPU else torch.device('cpu')

        self.inference_cache = InferenceCache()

        self.current_iteration = 0

        channels, rows, cols = CurrentGame.representation_shape
        self.hasher = ZobristHasherNumpy(channels, rows, cols)

    def load_model(self, model_path: str | PathLike) -> None:
        self.model = load_model(model_path, self.network_args, self.device)
        self.model.disable_auto_grad()
        self.model = self.model.eval()
        self.model.fuse_model()

    def update_iteration(self, iteration: int) -> None:
        """Update the Inference Client to use the model for the given iteration.
        Slighly optimizes the model for inference and resets the cache and stats."""

        self.current_iteration = iteration
        self.load_model(model_save_path(iteration, self.save_path))
        self.inference_cache.log_stats(iteration)
        self.inference_cache.clear_cache()

    def inference_batch(self, boards: list[np.ndarray]) -> list[tuple[MoveList, float]]:
        if not boards:
            return []

        encoded_boards = [decode_board_state(board) for board in boards]
        board_hashes = self.hasher.zobrist_hash_boards(np.array(encoded_boards))

        board_hashes_to_infer, boards_to_infer = self.inference_cache.filter(board_hashes, encoded_boards)

        if boards_to_infer:
            results = self._model_inference(boards_to_infer)

            for (hash, encoded_board), (policy, value) in zip(zip(board_hashes_to_infer, boards_to_infer), results):
                # Dont filter out the en passant moves here, because the policy is already fixed, but the en passant moves are dependent on the board state
                moves = filter_policy_with_en_passant_moves_then_get_moves_and_probabilities(
                    policy, CurrentGame.decode_canonical_board(encoded_board)
                )
                self.inference_cache.add(hash, np.array(moves), value)

        responses: list[tuple[MoveList, float]] = [self.inference_cache.get_encoded(hash) for hash in board_hashes]

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

    def check_for_iteration_update(self) -> None:
        iteration = get_latest_model_iteration(self.save_path)
        if iteration > self.current_iteration:
            self.update_iteration(iteration)


def _inference_server(device_id: int, server_address: str):
    server = BasicModelInference(
        device_id=device_id, network_args=TRAINING_ARGS.network, save_path=TRAINING_ARGS.save_path
    )

    context = zmq.Context.instance()
    # ROUTER socket to handle multiple client DEALER connections
    socket = context.socket(zmq.ROUTER)
    socket.bind(server_address)

    batch_inputs = []  # List of input data for the current batch.
    batch_meta = []  # List of tuples (client_identity, corr_id) for each input.
    batch_start_time = time.time()

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        # Poll with a timeout (in milliseconds)
        socks = dict(poller.poll(timeout=int(BATCH_TIMEOUT * 1000)))
        if socket in socks:
            # ROUTER sockets receive multipart messages:
            # [client_identity, empty_frame, payload]
            msg = socket.recv_multipart()
            client_id, payload = msg
            try:
                # Expect payload to be a list of dicts with keys "corr_id" and "input"
                for corr_id, inp in pickle.loads(payload):
                    batch_inputs.append(inp)
                    batch_meta.append((client_id, corr_id))
            except Exception as e:
                print('Error decoding client message:', e)
                continue

        # If we have some messages and either the batch is full or the timeout expired...
        if batch_inputs and (len(batch_inputs) >= BATCH_SIZE or (time.time() - batch_start_time) >= BATCH_TIMEOUT):
            server.check_for_iteration_update()
            # Perform batched inference
            results = server.inference_batch(batch_inputs)
            # Send each result back to the corresponding client with its correlation id.

            # group results by client_id
            results_by_client = defaultdict(list)
            for (client_id, corr_id), result in zip(batch_meta, results):
                results_by_client[client_id].append((corr_id, result))

            for client_id, results in results_by_client.items():
                payload = pickle.dumps(results)
                # The ROUTER socket requires the client identity and an empty frame.
                socket.send_multipart([client_id, payload])

            # Reset the batch and timer.
            batch_inputs = []
            batch_meta = []
            batch_start_time = time.time()


def run_inference_server(run: int | None, device_id: int, server_address: str):
    with log_exceptions(f'Inference Server {device_id} crashed.'):
        with TensorboardWriter(run, 'inference_server') if run is not None else contextlib.nullcontext():
            _inference_server(device_id, server_address)
