import time
import torch
import numpy as np
from typing import Callable
from dataclasses import dataclass
from torch.multiprocessing import Queue
from viztracer import VizTracer

from src.Encoding import decode_board_state
from src.Network import Network
from src.settings import TORCH_DTYPE, TRAINING_ARGS, USE_GPU, log_histogram, log_scalar, CurrentGame
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import load_model, model_save_path
from src.util.PipeConnection import PipeConnection


# TODO inference server handles with queues and is as light weight as possible
# TODO the load balancer will manage locations of incoming requests, caches, load balancing to the inference servers and proper redistribution of requests to the callers (in the correct order?)
# TODO remove caching from clients? Do they get the results in the correct order?
# TODO start a new parallel game as soon as a game finishes, not when all games finish? When to check for model/iteration updates?
# TODO optimize MCTS Node again

# Alternative approach
# TODO use asyncio to handle many games and search trees in parallel, assembling the requests into a batch and evaluating them locally
# TODO use a local asyncio event to notify once the network inference was done, so that results can be processed and the next iteration can be started
# Drawbacks:
# - less caching possible
# - GPU utilization based on how well the os schedules the processes
# - multiple models per GPU loaded - less vram remaining - lower batch size
# Benefits:
# - simpler architecture - actually more of a drawback if the project should be shown off
# - less communication overhead


def run_inference_server(
    input_queue: Queue,
    output_queue: Queue,
    commander_pipe: PipeConnection,
    device_id: int,
    timeout: float = 0.05,
):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable'
    assert TRAINING_ARGS.inference.batch_size > 0, 'Batch size must be greater than 0'
    assert timeout >= 0, 'Timeout must be non-negative'
    assert 0 <= device_id < torch.cuda.device_count() or not USE_GPU, f'Invalid device ID ({device_id})'

    server = InferenceServer(input_queue, output_queue, commander_pipe, device_id, timeout)
    with log_exceptions(f'Inference server ({device_id})'):
        server.run()


def start_inference_server(iteration: int) -> tuple[Queue, Queue, Callable[[], None]]:
    from multiprocessing import Pipe, Process

    input_queue, output_queue = Queue(), Queue()
    commander_pipe, commander_pipe_child = Pipe(duplex=False)

    inference_server = Process(target=run_inference_server, args=(input_queue, output_queue, commander_pipe, iteration))
    inference_server.start()

    commander_pipe_child.send(f'START AT ITERATION: {iteration}')

    def stop_inference_server():
        commander_pipe_child.send('STOP')
        inference_server.join()

    return input_queue, output_queue, stop_inference_server
    # return InferenceClient(input_queue, output_queue), stop_inference_server


inference_requests = 0
inference_calc_time = 0


@dataclass
class RequestBatch:
    request_id: bytes
    game_state: np.ndarray


BoardHash = bytes


def run_caching_layer(
    input_queue: Queue,
    output_queues: list[Queue],
    process_input_queue: Queue,
    process_output_queue: Queue,
    commander_pipe: PipeConnection,
):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable'

    layer = CachingLayer(input_queue, output_queues, process_input_queue, process_output_queue, commander_pipe)
    with log_exceptions('Caching layer'):
        layer.run()


class CachingLayer:
    def __init__(
        self,
        input_queue: Queue,
        output_queues: list[Queue],
        process_input_queue: Queue,
        process_output_queue: Queue,
        commander_pipe: PipeConnection,
    ):
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.process_input_queue = process_input_queue
        self.process_output_queue = process_output_queue
        self.commander_pipe = commander_pipe

        self.cache: dict[BoardHash, bytes] = {}
        self.active_requests: dict[tuple[bytes, int], BoardHash] = {}

        self.total_evals = 0
        self.total_hits = 0

    def run(self):
        while True:
            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    self._clear_cache(current_iteration)

            for _ in range(1000):  # Only process the commander pipe every 1000 iterations
                if not self.input_queue.empty():
                    message = self.input_queue.get()
                    assert isinstance(message, bytes), f'Expected message to be bytes, got {message}'
                    self._process_request(message)

                if not self.process_output_queue.empty():
                    message = self.process_output_queue.get()
                    assert isinstance(message, bytes), f'Expected message to be bytes, got {message}'
                    self._process_response(message)

    def _process_request(self, message: bytes) -> None:
        request_id, sender_id, board_hash = self._decode_message(message)

        if board_hash in self.cache:
            return self._send_response(request_id, sender_id, board_hash)

        self.active_requests[(request_id, sender_id)] = board_hash
        self.process_input_queue.put_nowait(message)

    def _process_response(self, message: bytes) -> None:
        request_id, sender_id, result = self._decode_message(message)

        board_hash = self.active_requests.pop((request_id, sender_id))

        self.cache[board_hash] = result

        self._send_response(request_id, sender_id, board_hash)

    def _send_response(self, request_id: bytes, sender_id: int, board_hash: BoardHash) -> None:
        result = self.cache[board_hash]

        self.output_queues[sender_id].put_nowait(request_id + result)

    def _decode_message(self, message: bytes) -> tuple[bytes, int, bytes]:
        request_id = message[:4]
        sender_id = int.from_bytes(message[4:6], 'big')
        data = message[6:]

        assert 0 <= sender_id < len(self.output_queues), f'Invalid sender ID ({sender_id})'

        return request_id, sender_id, data

    def _clear_cache(self, iteration: int) -> None:
        if self.total_evals != 0:
            cache_hit_rate = (self.total_hits / self.total_evals) * 100
            log_scalar('cache_hit_rate', cache_hit_rate, iteration)
            log_scalar('unique_positions_in_cache', len(self.cache), iteration)
            # log_histogram(
            #     'nn_output_value_distribution',
            #     np.array([round(v.item(), 1) for _, v in self.cache.values()]),
            #     iteration,
            # )

            log(f'Cache hit rate: {cache_hit_rate:.2f}% on cache size {len(self.cache)}')
        self.cache.clear()


class InferenceServer:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        commander_pipe: PipeConnection,
        device_id: int,
        timeout: float,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.commander_pipe = commander_pipe
        self.timeout = timeout

        self.model: Network = None  # type: ignore
        self.device = torch.device(f'cuda:{device_id}') if USE_GPU else torch.device('cpu')

    def run(self):
        tracer = VizTracer(max_stack_depth=11)
        tracer.start()

        while True:
            for _ in range(1000):  # Only process the commander pipe every 1000 iterations
                if not self.input_queue.empty() and self.model is not None:
                    self._process_batch(*self._get_batch_requests())

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    tracer.stop()
                    tracer.save(f'inference_trace_{self.device}.json')
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    self._reload_model(current_iteration)

        log(f'Inference server {self.device} stopped')

    def _reload_model(self, iteration: int) -> None:
        self.model = load_model(model_save_path(iteration), TRAINING_ARGS.network, self.device)
        self.model = self._prepare_model_for_inference(self.model)

    def _get_batch_requests(self) -> tuple[list[bytes], list[np.ndarray]]:
        request_ids: list[bytes] = []
        board_states: list[np.ndarray] = []
        batch_start_time = time.time()

        while time.time() - batch_start_time < self.timeout:
            if not self.input_queue.empty():
                request_id, board_state = self._get_batch_request()

                request_ids.append(request_id)
                board_states.append(board_state)

                if len(board_states) >= TRAINING_ARGS.inference.batch_size:
                    log('Batch full, processing...')
                    break
        else:
            if len(board_states):
                log(f'Batch timeout, processing {len(board_states)} inferences from as many total requests...')

        return request_ids, board_states

    def _get_batch_request(self) -> tuple[bytes, np.ndarray]:
        message = self.input_queue.get()
        channels = CurrentGame.representation_shape[0]

        encoded_board = np.frombuffer(message[6:], dtype=np.uint64).reshape(-1, channels)

        return message[:6], decode_board_state(encoded_board)

    def _prepare_model_for_inference(self, model: Network) -> Network:
        model.eval()
        # TODO int8 quantized model
        return model

    @torch.no_grad()
    def _process_batch(self, request_ids: list[bytes], board_states: list[np.ndarray]) -> None:
        if not board_states:
            return

        global inference_requests, inference_calc_time
        inference_requests += 1
        start = time.time()

        input_tensor = torch.tensor(np.array(board_states), dtype=TORCH_DTYPE).to(self.model.device)

        policies, values = self.model(input_tensor)

        # TODO Wait for the model to finish the inference and simultaneously prefetch the next batch

        policies = torch.softmax(policies, dim=1).to(dtype=torch.float32, device='cpu').numpy()
        values = torch.mean(values, dim=1).to(dtype=torch.float32, device='cpu').numpy()

        inf_calc_time = time.time() - start
        inference_calc_time += inf_calc_time

        log(f'Average inference calc time: {inf_calc_time} on average: {inference_calc_time / inference_requests:.2f}s')

        for request_id, policy, value in zip(request_ids, policies, values):
            self._send_response(request_id, policy, value)

    def _send_response(self, request_id: bytes, policies: np.ndarray, values: np.ndarray) -> None:
        self.output_queue.put_nowait(request_id + np.concatenate((policies, [values])).tobytes())
