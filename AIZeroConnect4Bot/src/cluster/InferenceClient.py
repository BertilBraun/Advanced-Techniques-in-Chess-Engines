import time
import asyncio
import numpy as np
from src.settings import CurrentBoard, CurrentGame
from src.Encoding import encode_board_state
from src.util.PipeConnection import PipeConnection
from src.util.log import log

inference_calls = -50
inference_time = 0

_INFERENCE_RESPONSES: dict[bytes, bytes | None] = {}


class InferenceClient:
    def __init__(self, server_conn: PipeConnection):
        assert server_conn.readable and server_conn.writable, 'PipeConnection must be readable and writable'
        self.server_conn = server_conn

    async def inference(self, boards: list[CurrentBoard]) -> tuple[np.ndarray, np.ndarray]:
        encoded_boards = [encode_board_state(CurrentGame.get_canonical_board(board)) for board in boards]
        encoded_bytes = np.array(encoded_boards).tobytes()

        global inference_calls, inference_time
        inference_calls += 1
        start = time.time()

        self.server_conn.send_bytes(encoded_bytes)

        _INFERENCE_RESPONSES[encoded_bytes] = None

        # TODO if that is done async, then the recieved bytes could be from a different send request in another iteration of the inference loop. Somehow they will have to be matched back up

        await asyncio.sleep(0.001)  # About the min time it takes the inference server to process a request
        while not _INFERENCE_RESPONSES[encoded_bytes]:
            if self.server_conn.poll():
                hashes_and_results = self.server_conn.recv_bytes()
                hashes, results = hashes_and_results.split(b'\n\n\n')
                _INFERENCE_RESPONSES[hashes] = results
            else:
                await asyncio.sleep(0.00001)

        result = _INFERENCE_RESPONSES[encoded_bytes]
        assert result is not None, 'Result should not be None'

        inference_time += time.time() - start

        if inference_calls % 10000 == 1:
            pass  # log(f'Average inference request time: {inference_time / inference_calls:.2f}s')
        if inference_calls == 0:
            inference_time = 0

        result = np.frombuffer(result, dtype=np.float32).reshape(-1, CurrentGame.action_size + 1)
        policy, value = result[:, :-1], result[:, -1]

        return policy, value
