import time
import asyncio_pipe
import numpy as np
from src.settings import CurrentBoard, CurrentGame
from src.Encoding import encode_board_state
from src.util.PipeConnection import PipeConnection
from src.util.log import log

inference_calls = -50
inference_time = 0


class InferenceClient:
    def __init__(self, server_conn: PipeConnection):
        assert server_conn.readable and server_conn.writable, 'PipeConnection must be readable and writable'
        self.server_conn = asyncio_pipe.Connection(server_conn)

    async def inference(self, boards: list[CurrentBoard]) -> tuple[np.ndarray, np.ndarray]:
        encoded_boards = [encode_board_state(CurrentGame.get_canonical_board(board)) for board in boards]
        encoded_bytes = np.array(encoded_boards).tobytes()

        global inference_calls, inference_time
        inference_calls += 1
        start = time.time()

        self.server_conn.send_bytes(encoded_bytes)

        result = await self.server_conn.recv_bytes()

        inference_time += time.time() - start

        if inference_calls % 100 == 0:
            log(f'Average inference request time: {inference_time / inference_calls:.2f}s')
        if inference_calls == 0:
            inference_time = 0

        result = np.frombuffer(result, dtype=np.float32).reshape(-1, CurrentGame.action_size + 1)
        policy, value = result[:, :-1], result[:, -1]

        return policy, value
