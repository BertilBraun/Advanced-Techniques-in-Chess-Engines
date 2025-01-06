import numpy as np
from multiprocessing.connection import PipeConnection
from src.settings import CurrentBoard, CurrentGame
from src.Encoding import encode_board_state


class InferenceClient:
    def __init__(self, server_conn: PipeConnection):
        assert server_conn.readable and server_conn.writable, 'Connection must be readable and writable'
        self.server_conn = server_conn

    def inference(self, boards: list[CurrentBoard]) -> tuple[np.ndarray, np.ndarray]:
        encoded_boards = [encode_board_state(CurrentGame.get_canonical_board(board)) for board in boards]
        self.server_conn.send(np.array(encoded_boards))

        policy, value = np.zeros((len(boards), CurrentGame.action_size)), np.zeros(len(boards))

        for i in range(len(boards)):
            policy[i], value[i] = self.server_conn.recv()

        return policy, value
