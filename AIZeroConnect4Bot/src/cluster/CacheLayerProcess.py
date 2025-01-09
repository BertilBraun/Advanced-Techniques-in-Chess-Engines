from torch.multiprocessing import Queue

from src.settings import log_scalar
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.PipeConnection import PipeConnection


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
        self.active_requests: dict[tuple[int, int], BoardHash] = {}

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
                    self._process_request(self.input_queue.get())

                if not self.process_output_queue.empty():
                    self._process_response(self.process_output_queue.get())

    def _process_request(self, message: tuple[int, int, bytes]) -> None:
        request_id, sender_id, board_hash = message

        if board_hash in self.cache:
            return self._send_response(request_id, sender_id, self.cache[board_hash])

        self.active_requests[(request_id, sender_id)] = board_hash
        self.process_input_queue.put_nowait(message)

    def _process_response(self, message: tuple[int, int, bytes]) -> None:
        request_id, sender_id, result = message

        board_hash = self.active_requests.pop((request_id, sender_id))

        self.cache[board_hash] = result

        self._send_response(request_id, sender_id, result)

    def _send_response(self, request_id: int, sender_id: int, result: bytes) -> None:
        self.output_queues[sender_id].put_nowait((request_id, result))

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
