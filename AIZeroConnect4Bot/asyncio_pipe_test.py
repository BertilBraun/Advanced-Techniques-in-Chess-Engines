import asyncio
import multiprocessing

import numpy as np

from src.Encoding import encode_board_state
from src.cluster.InferenceServerProcess import start_inference_server

from src.settings import CurrentGame


if __name__ == '__main__':
    input_q, output_q, stop = start_inference_server(0)
    print('Creating client')

    num_inferences = 0
    BATCH_SIZE = 300

    boards = np.random.randint(0, 2, (BATCH_SIZE, *CurrentGame.representation_shape))
    encoded_boards = [encode_board_state(state) for state in boards]
    encoded_bytes = np.array(encoded_boards).tobytes()
    input_q.put((0).to_bytes(4, 'big') + encoded_bytes)

    for i in range(1000000):
        if not output_q.empty():
            output_q.get()
            num_inferences += BATCH_SIZE
        boards = np.random.randint(0, 2, (BATCH_SIZE, *CurrentGame.representation_shape))
        encoded_boards = [encode_board_state(state) for state in boards]
        encoded_bytes = np.array(encoded_boards).tobytes()
        input_q.put((i).to_bytes(4, 'big') + encoded_bytes)
