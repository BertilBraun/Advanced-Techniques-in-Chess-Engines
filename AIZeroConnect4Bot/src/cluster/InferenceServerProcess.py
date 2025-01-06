import time
import torch
import numpy as np
from typing import Callable
from multiprocessing import Process
from multiprocessing.connection import Pipe, PipeConnection

from src.Encoding import decode_board_state
from src.Network import Network
from src.cluster.InferenceClient import InferenceClient
from src.settings import TORCH_DTYPE, TRAINING_ARGS, USE_GPU, log_histogram, log_scalar, CurrentGame
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import create_model, load_model, model_save_path


def run_inference_server(
    inference_input_pipe: PipeConnection,
    commander_pipe: PipeConnection,
    device_id: int,
    timeout: float = 0.05,
):
    assert inference_input_pipe.readable and inference_input_pipe.writable, 'Input pipe must be readable and writable'
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable'
    assert TRAINING_ARGS.inference.batch_size > 0, 'Batch size must be greater than 0'
    assert timeout >= 0, 'Timeout must be non-negative'
    assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, 'Invalid device ID'

    server = InferenceServer(inference_input_pipe, commander_pipe, device_id, timeout)
    with log_exceptions(f'Inference server ({device_id})'):
        server.run()


def start_inference_server(iteration: int) -> tuple[InferenceClient, Callable[[], None]]:
    inference_server_pipe, inference_client_pipe = Pipe()
    commander_pipe, commander_pipe_child = Pipe()

    inference_server = Process(target=run_inference_server, args=(inference_server_pipe, commander_pipe, iteration))
    inference_server.start()

    commander_pipe_child.send(f'START AT ITERATION: {iteration}')

    def stop_inference_server():
        commander_pipe_child.send('STOP')
        inference_server.join()

    return InferenceClient(inference_client_pipe), stop_inference_server


class InferenceServer:
    def __init__(
        self,
        inference_input_pipe: PipeConnection,
        commander_pipe: PipeConnection,
        device_id: int,
        timeout: float = 0.05,
    ):
        self.inference_input_pipe = inference_input_pipe  # Pipe connection
        self.commander_pipe = commander_pipe
        self.timeout = timeout

        self.device = torch.device(f'cuda:{device_id}') if USE_GPU else torch.device('cpu')

        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.total_evals = 0
        self.total_hits = 0

    def run(self):
        num_non_cached_requests = 0
        batch_requests: list[list[tuple[bytes, np.ndarray | None]]] = []
        time_last_batch = time.time()

        self.model = create_model(TRAINING_ARGS.network, self.device)
        self.model = self._prepare_model_for_inference(self.model)

        while True:
            # Check for new requests with a timeout
            if self.inference_input_pipe.poll(self.timeout):
                message = self.inference_input_pipe.recv_bytes()
                channels = CurrentGame.representation_shape[0]
                encoded_boards = np.frombuffer(message, dtype=np.uint64).reshape(-1, channels)

                request_batch: list[tuple[bytes, np.ndarray | None]] = []
                hashes = [encoded_board.data.tobytes() for encoded_board in encoded_boards]
                for i, (board_hash, encoded_board) in enumerate(zip(hashes, encoded_boards)):
                    if board_hash not in self.cache and board_hash not in hashes[:i]:
                        num_non_cached_requests += 1
                        decoded_board = decode_board_state(encoded_board)
                    else:
                        decoded_board = None
                    request_batch.append((board_hash, decoded_board))

                batch_requests.append(request_batch)

                time_last_batch = time.time()

                if num_non_cached_requests >= TRAINING_ARGS.inference.batch_size:
                    self._process_batch(batch_requests)
                    batch_requests = []
                    num_non_cached_requests = 0
                    time_last_batch = time.time()

            # Check if the oldest request has been waiting longer than the timeout
            if batch_requests and (time.time() - time_last_batch) >= self.timeout:
                self._process_batch(batch_requests)
                batch_requests = []
                num_non_cached_requests = 0
                time_last_batch = time.time()

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    self.model = load_model(model_save_path(current_iteration), TRAINING_ARGS.network, self.device)
                    self.model = self._prepare_model_for_inference(self.model)
                    self._clear_cache(current_iteration)

        log(f'Inference server {self.device} stopped')

    def _clear_cache(self, iteration: int) -> None:
        if self.total_evals != 0:
            log_scalar('cache_hit_rate', self.total_hits / self.total_evals, iteration)
            log_scalar('unique_positions_in_cache', len(self.cache), iteration)
            log_histogram(
                'nn_output_value_distribution',
                np.array([round(v.item(), 1) for _, v in self.cache.values()]),
                iteration,
            )

            log(f'Cache hit rate: {self.total_hits / self.total_evals} on cache size {len(self.cache)}')
        self.cache.clear()

    def _prepare_model_for_inference(self, model: Network) -> Network:
        model.eval()
        # TODO int8 quantized model
        return model

    @torch.no_grad()
    def _process_batch(self, batch_requests: list[list[tuple[bytes, np.ndarray | None]]]) -> None:
        to_process: list[np.ndarray] = []
        to_process_hashes: list[bytes] = []
        for request_batch in batch_requests:
            for hash, board in request_batch:
                if board is not None:
                    to_process.append(board)
                    to_process_hashes.append(hash)

        if to_process:
            input_tensor = torch.tensor(np.array(to_process), dtype=TORCH_DTYPE).to(self.model.device)

            policies, values = self.model(input_tensor)

            policies = torch.softmax(policies, dim=1).to(dtype=torch.float32, device='cpu').numpy()
            values = values.to(dtype=torch.float32, device='cpu').numpy().mean(axis=1)

            for hash, p, v in zip(to_process_hashes, policies, values):
                self.cache[hash] = (p.copy(), v.copy())

        total_samples_evaluated = sum(len(batch) for batch in batch_requests)
        self.total_evals += total_samples_evaluated
        self.total_hits += total_samples_evaluated - len(to_process)

        for request_batch in batch_requests:
            batch_results: list[np.ndarray] = []

            for hash, _ in request_batch:
                policy, value = self.cache[hash]
                batch_results.append(np.concatenate((policy, [value])))

            self.inference_input_pipe.send_bytes(np.array(batch_results).tobytes())
