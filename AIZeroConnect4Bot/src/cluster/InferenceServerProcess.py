import time
import torch
import numpy as np
from typing import Callable
from torch.multiprocessing import Process, Pipe

from src.Encoding import decode_board_state
from src.Network import Network
from src.cluster.InferenceClient import InferenceClient
from src.settings import TORCH_DTYPE, TRAINING_ARGS, USE_GPU, log_histogram, log_scalar, CurrentGame
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import load_model, model_save_path
from src.util.PipeConnection import PipeConnection


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
    assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, f'Invalid device ID ({device_id})'

    server = InferenceServer(inference_input_pipe, commander_pipe, device_id, timeout)
    with log_exceptions(f'Inference server ({device_id})'):
        server.run()


def start_inference_server(iteration: int) -> tuple[InferenceClient, Callable[[], None]]:
    inference_server_pipe, inference_client_pipe = Pipe()
    commander_pipe, commander_pipe_child = Pipe(duplex=False)

    inference_server = Process(target=run_inference_server, args=(inference_server_pipe, commander_pipe, iteration))
    inference_server.start()

    commander_pipe_child.send(f'START AT ITERATION: {iteration}')

    def stop_inference_server():
        commander_pipe_child.send('STOP')
        inference_server.join()

    return InferenceClient(inference_client_pipe), stop_inference_server


inference_requests = 0
inference_calc_time = 0


class InferenceServer:
    def __init__(
        self,
        inference_input_pipe: PipeConnection,
        commander_pipe: PipeConnection,
        device_id: int,
        timeout: float = 0.05,
    ):
        self.inference_input_pipe = inference_input_pipe
        self.commander_pipe = commander_pipe
        self.timeout = timeout

        self.model: Network = None  # type: ignore
        self.device = torch.device(f'cuda:{device_id}') if USE_GPU else torch.device('cpu')

        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.total_evals = 0
        self.total_hits = 0

    def run(self):
        while True:
            if self.inference_input_pipe.poll(self.timeout) and self.model is not None:
                batch_requests = self._get_batch_requests()
                if batch_requests:
                    self._process_batch(batch_requests)

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    self._reload_model(current_iteration)

        log(f'Inference server {self.device} stopped')

    def _reload_model(self, iteration: int) -> None:
        self.model = load_model(model_save_path(iteration), TRAINING_ARGS.network, self.device)
        self.model = self._prepare_model_for_inference(self.model)
        self._clear_cache(iteration)

    def _get_batch_requests(self) -> list[list[tuple[bytes, np.ndarray | None]]]:
        batch_requests: list[list[tuple[bytes, np.ndarray | None]]] = []
        all_hashes: set[bytes] = set()
        batch_start_time = time.time()

        while time.time() - batch_start_time < self.timeout:
            if self.inference_input_pipe.poll(self.timeout):
                request_batch, batch_new_hashes, batch_required_hashes = self._get_batch_request(all_hashes)

                # If no new hashes and there is not a single hash in the required hashes that is also in all_hashes, i.e. the intersection is empty
                if len(batch_new_hashes) == 0 and batch_required_hashes.isdisjoint(all_hashes):
                    self._send_response_from_cache(request_batch)
                else:
                    batch_requests.append(request_batch)

                all_hashes.update(batch_new_hashes)

                if len(all_hashes) >= TRAINING_ARGS.inference.batch_size:
                    pass  # log('Batch full, processing...')
                    break
        else:
            if len(batch_requests):
                pass  # log(
                pass  #     f'Batch timeout, processing {len(all_hashes)} inferences from {len(batch_requests)} total requests...'
                pass  # )

        return batch_requests

    def _get_batch_request(
        self, all_hashes: set[bytes]
    ) -> tuple[list[tuple[bytes, np.ndarray | None]], set[bytes], set[bytes]]:
        message = self.inference_input_pipe.recv_bytes()
        channels = CurrentGame.representation_shape[0]
        encoded_boards = np.frombuffer(message, dtype=np.uint64).reshape(-1, channels)

        hashes: list[bytes] = [encoded_board.data.tobytes() for encoded_board in encoded_boards]

        my_new_hashes: set[bytes] = set()
        my_required_hashes: set[bytes] = set()

        request_batch: list[tuple[bytes, np.ndarray | None]] = []
        for board_hash, encoded_board in zip(hashes, encoded_boards):
            if board_hash not in self.cache and board_hash not in all_hashes and board_hash not in my_new_hashes:
                my_new_hashes.add(board_hash)
                decoded_board = decode_board_state(encoded_board)
            else:
                my_required_hashes.add(board_hash)
                decoded_board = None
            request_batch.append((board_hash, decoded_board))

        return request_batch, my_new_hashes, my_required_hashes

    def _clear_cache(self, iteration: int) -> None:
        if self.total_evals != 0:
            cache_hit_rate = (self.total_hits / self.total_evals) * 100
            log_scalar('cache_hit_rate', cache_hit_rate, iteration)
            log_scalar('unique_positions_in_cache', len(self.cache), iteration)
            log_histogram(
                'nn_output_value_distribution',
                np.array([round(v.item(), 1) for _, v in self.cache.values()]),
                iteration,
            )

            log(f'Cache hit rate: {cache_hit_rate:.2f}% on cache size {len(self.cache)}')
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
            global inference_requests, inference_calc_time
            inference_requests += 1
            start = time.time()

            input_tensor = torch.tensor(np.array(to_process), dtype=TORCH_DTYPE).to(self.model.device)

            policies, values = self.model(input_tensor)

            policies = torch.softmax(policies, dim=1).to(dtype=torch.float32, device='cpu').numpy()
            values = values.to(dtype=torch.float32, device='cpu').numpy().mean(axis=1)

            inf_calc_time = time.time() - start
            inference_calc_time += inf_calc_time

            pass  # log(
            pass  #     f'Average inference calc time: {inf_calc_time} on average: {inference_calc_time / inference_requests:.2f}s'
            pass  # )

            for hash, p, v in zip(to_process_hashes, policies, values):
                self.cache[hash] = (p.copy(), v.copy())

        total_samples_evaluated = sum(len(batch) for batch in batch_requests)
        self.total_evals += total_samples_evaluated
        self.total_hits += total_samples_evaluated - len(to_process)

        for request_batch in batch_requests:
            self._send_response_from_cache(request_batch)

    def _send_response_from_cache(self, request_batch: list[tuple[bytes, np.ndarray | None]]) -> None:
        batch_results: list[np.ndarray] = []

        for hash, _ in request_batch:
            policy, value = self.cache[hash]
            batch_results.append(np.concatenate((policy, [value])))

        # Send back the hash as well as the result concatenated
        result = np.array(batch_results).tobytes()
        hashes = b''.join(hash for hash, _ in request_batch)
        data = hashes + b'\n\n\n' + result
        self.inference_input_pipe.send_bytes(data)
