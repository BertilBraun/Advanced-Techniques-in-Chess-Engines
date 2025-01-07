import time
import numpy as np
from src.settings import CurrentBoard, CurrentGame
from src.Encoding import encode_board_state
from src.util.PipeConnection import PipeConnection
from src.util.log import log

inference_calls = -50
inference_time = 0
import asyncio


class AsyncConnection:
    def __init__(self, connection):
        self._connection = connection

    def send(self, obj):
        """Send a (picklable) object"""

        self._connection.send(obj)

    async def _wait_for_input(self):
        """Wait until there is an input available to be read"""
        print('Waiting for input')
        while not self._connection.poll():
            await asyncio.sleep(0.00001)
        print('Input available')

    async def recv(self):
        """Receive a (picklable) object"""

        await self._wait_for_input()
        return self._connection.recv()

    def fileno(self):
        """File descriptor or handle of the connection"""
        return self._connection.fileno()

    def close(self):
        """Close the connection"""
        self._connection.close()

    async def poll(self, timeout=0.0):
        """Whether there is an input available to be read"""

        if self._connection.poll():
            return True

        try:
            await asyncio.wait_for(self._wait_for_input(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return self._connection.poll()

    def send_bytes(self, buf, offset=0, size=None):
        """Send the bytes data from a bytes-like object"""

        self._connection.send_bytes(buf, offset, size)

    async def recv_bytes(self, maxlength=None):
        """
        Receive bytes data as a bytes object.
        """

        print('Receiving bytes')
        await self._wait_for_input()
        print('Received bytes')
        return self._connection.recv_bytes(maxlength)

    async def recv_bytes_into(self, buf, offset=0):
        """
        Receive bytes data into a writeable bytes-like object.
        Return the number of bytes read.
        """

        await self._wait_for_input()
        return self._connection.recv_bytes_into(buf, offset)


class InferenceClient:
    def __init__(self, server_conn: PipeConnection):
        assert server_conn.readable and server_conn.writable, 'PipeConnection must be readable and writable'
        self.server_conn = AsyncConnection(server_conn)

    async def inference(self, boards: list[CurrentBoard]) -> tuple[np.ndarray, np.ndarray]:
        encoded_boards = [encode_board_state(CurrentGame.get_canonical_board(board)) for board in boards]
        encoded_bytes = np.array(encoded_boards).tobytes()

        global inference_calls, inference_time
        inference_calls += 1
        start = time.time()

        self.server_conn.send_bytes(encoded_bytes)

        result = await self.server_conn.recv_bytes()

        inference_time += time.time() - start

        if inference_calls % 100 == 1:
            log(f'Average inference request time: {inference_time / inference_calls:.2f}s')
        if inference_calls == 0:
            inference_time = 0

        result = np.frombuffer(result, dtype=np.float32).reshape(-1, CurrentGame.action_size + 1)
        policy, value = result[:, :-1], result[:, -1]

        return policy, value
