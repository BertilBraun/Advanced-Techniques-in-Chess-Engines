import time
import torch
import numpy as np
from typing import Callable
from torch.multiprocessing import Queue

from src.Encoding import decode_board_state
from src.Network import Network
from src.cluster.InferenceClient import InferenceClient
from src.settings import TORCH_DTYPE, TRAINING_ARGS, USE_GPU, CurrentGame
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import load_model, model_save_path
from src.util.PipeConnection import PipeConnection


# DONE inference server handles with queues and is as light weight as possible
# DONE the load balancer will manage locations of incoming requests, caches, load balancing to the inference servers and proper redistribution of requests to the callers (in the correct order?)
# NOT_NECESSARY remove caching from clients? Do they get the results in the correct order?
# DONE start a new parallel game as soon as a game finishes, not when all games finish? When to check for model/iteration updates?
# DONE optimize MCTS Node again
# TODO batch on the cache layer already and send complete batches to the inference server

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


def start_inference_server(iteration: int) -> tuple[InferenceClient, Callable[[], None]]:
    from torch.multiprocessing import Pipe, Process

    input_queue, output_queue = Queue(), Queue()
    commander_pipe, commander_pipe_child = Pipe(duplex=False)

    inference_server = Process(target=run_inference_server, args=(input_queue, output_queue, commander_pipe, 0))
    inference_server.start()

    commander_pipe_child.send(f'START AT ITERATION: {iteration}')

    def stop_inference_server():
        commander_pipe_child.send('STOP')
        inference_server.join()

    return InferenceClient(input_queue, output_queue, -1), stop_inference_server


RequestId = tuple[int, int]


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
        while True:
            for _ in range(1000):  # Only process the commander pipe every 1000 iterations
                if not self.input_queue.empty() and self.model is not None:
                    self._process_batch(*self._get_batch_requests())

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

    def _get_batch_requests(self) -> tuple[list[RequestId], list[np.ndarray]]:
        request_ids: list[RequestId] = []
        board_states: list[np.ndarray] = []
        batch_start_time = time.time()

        while time.time() - batch_start_time < self.timeout:
            if not self.input_queue.empty():
                request_id, board_state = self._get_batch_request()

                request_ids.append(request_id)
                board_states.append(board_state)

                if len(board_states) >= TRAINING_ARGS.inference.batch_size:
                    break

        return request_ids, board_states

    def _get_batch_request(self) -> tuple[RequestId, np.ndarray]:
        request_id, sender_id, encoded_bytes = self.input_queue.get()
        channels = CurrentGame.representation_shape[0]

        encoded_board = np.frombuffer(encoded_bytes, dtype=np.uint64).reshape(1, channels)

        return (request_id, sender_id), decode_board_state(encoded_board)

    def _prepare_model_for_inference(self, model: Network) -> Network:
        model.eval()
        # TODO int8 quantized model
        return model

    @torch.no_grad()
    def _process_batch(self, request_ids: list[RequestId], board_states: list[np.ndarray]) -> None:
        if not board_states:
            return

        input_tensor = torch.tensor(np.array(board_states), dtype=TORCH_DTYPE, device=self.model.device)

        policies, values = self.model(input_tensor)

        # TODO Wait for the model to finish the inference and simultaneously prefetch the next batch

        policies = torch.softmax(policies, dim=1)
        values = torch.mean(values, dim=1)

        results = torch.cat((policies, values.unsqueeze(1)), dim=1)  # TODO check correctness

        # Do as much as possible in parallel on the GPU before sending the results back to CPU
        results = results.to(dtype=torch.float32, device='cpu').numpy()

        for request_id, result in zip(request_ids, results):
            self._send_response(request_id, result)

    def _send_response(self, request: RequestId, result: np.ndarray) -> None:
        request_id, sender_id = request
        self.output_queue.put_nowait((request_id, sender_id, result.tobytes()))
