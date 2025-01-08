from collections import defaultdict
import time
import asyncio
import numpy as np
from src.settings import CurrentBoard, CurrentGame
from src.Encoding import encode_board_state
from src.util.PipeConnection import PipeConnection
from src.util.log import log

inference_calls = -50
inference_time = 0

_INFERENCE_RESPONSES: dict[int, bytes] = defaultdict(bytes)
_INFERENCE_REQUESTS: list[bytes] = []
_NUM_BOARDS_IN_INFERENCE_REQUEST = defaultdict(int)
_REQUEST_INDEX = 0


class InferenceClient:
    def __init__(self, server_conn: PipeConnection):
        assert server_conn.readable and server_conn.writable, 'PipeConnection must be readable and writable'
        self.server_conn = server_conn

    async def inference(self, boards: list[CurrentBoard]) -> tuple[np.ndarray, np.ndarray]:
        global _INFERENCE_RESPONSES, _INFERENCE_REQUESTS, _NUM_BOARDS_IN_INFERENCE_REQUEST, _REQUEST_INDEX

        encoded_boards = [encode_board_state(CurrentGame.get_canonical_board(board)) for board in boards]
        encoded_bytes = np.array(encoded_boards).tobytes()

        my_request_index = _REQUEST_INDEX
        my_index = _NUM_BOARDS_IN_INFERENCE_REQUEST[my_request_index]
        _NUM_BOARDS_IN_INFERENCE_REQUEST[my_request_index] += len(boards)
        _INFERENCE_REQUESTS.append(encoded_bytes)

        global inference_calls, inference_time
        inference_calls += 1
        start = time.time()

        while (
            _NUM_BOARDS_IN_INFERENCE_REQUEST[my_request_index] < 100
            and time.time() - start < 0.01
            and _REQUEST_INDEX == my_request_index
        ):
            await asyncio.sleep(0.001)

        if _REQUEST_INDEX == my_request_index:
            # send request index and the joined bytes of all the requests
            self.server_conn.send_bytes(_REQUEST_INDEX.to_bytes(4, 'big') + b''.join(_INFERENCE_REQUESTS))
            _INFERENCE_REQUESTS = []
            _REQUEST_INDEX += 1

        # If that is done async, then the recieved bytes could be from a different send request in another iteration of the inference loop. Somehow they will have to be matched back up

        while not _INFERENCE_RESPONSES[my_request_index]:
            if self.server_conn.poll():
                results = self.server_conn.recv_bytes()
                request_id = int.from_bytes(results[:4], 'big')
                _INFERENCE_RESPONSES[request_id] = results[4:]

                if request_id - 50 in _INFERENCE_RESPONSES:
                    # Clean up the memory to prevent memory leaks
                    del _INFERENCE_RESPONSES[request_id - 50]
                    del _NUM_BOARDS_IN_INFERENCE_REQUEST[request_id - 50]

            else:
                await asyncio.sleep(0.00001)

        inference_time += time.time() - start

        if inference_calls % 10000 == 1:
            pass  # log(f'Average inference request time: {inference_time / inference_calls:.2f}s')
        if inference_calls == 0:
            inference_time = 0

        stride = (CurrentGame.action_size + 1) * 4

        start = my_index * stride
        end = (my_index + len(boards)) * stride
        result = _INFERENCE_RESPONSES[my_request_index][start:end]

        result = np.frombuffer(result, dtype=np.float32).reshape(-1, CurrentGame.action_size + 1)
        policy, value = result[:, :-1], result[:, -1]

        return policy, value
