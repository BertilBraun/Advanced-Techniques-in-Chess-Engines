from __future__ import annotations

import asyncio
import numpy as np
from queue import Empty
from torch.multiprocessing import Queue
from collections import defaultdict

from src.Encoding import encode_board_state
from src.settings import CurrentBoard, CurrentGame

_REQUEST_INDEX = 0
_INFERENCE_RESPONSES: dict[int, bytes] = defaultdict(bytes)

_RESULT_CACHE: dict[bytes, tuple[np.ndarray, float]] = {}


class InferenceClient:
    def __init__(self, inference_queue: Queue, result_queue: Queue, global_id: int) -> None:
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.global_id = global_id

    def reset_cache(self):
        _RESULT_CACHE.clear()

    async def inference(self, board: CurrentBoard) -> tuple[np.ndarray, float]:
        global _INFERENCE_RESPONSES, _REQUEST_INDEX

        encoded_board = encode_board_state(CurrentGame.get_canonical_board(board))
        encoded_bytes = np.array(encoded_board).tobytes()

        if encoded_bytes in _RESULT_CACHE:
            return _RESULT_CACHE[encoded_bytes]

        my_request_index = _REQUEST_INDEX
        _REQUEST_INDEX = (_REQUEST_INDEX + 1) % (2**32)

        self.inference_queue.put((my_request_index, self.global_id, encoded_bytes))

        # If that is done async, then the recieved bytes could be from a different send request in another iteration of the inference loop. Somehow they will have to be matched back up

        while not _INFERENCE_RESPONSES[my_request_index]:
            try:
                request_id, results = self.result_queue.get_nowait()
                _INFERENCE_RESPONSES[request_id] = results
            except Empty:
                await asyncio.sleep(0.005)

        result = _INFERENCE_RESPONSES.pop(my_request_index)

        result = np.frombuffer(result, dtype=np.float32).reshape(CurrentGame.action_size + 1)
        policy, value = result[:-1], result[-1]

        _RESULT_CACHE[encoded_bytes] = policy, value

        return policy, value
