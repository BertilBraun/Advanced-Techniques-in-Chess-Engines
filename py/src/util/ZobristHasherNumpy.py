import numpy as np


class ZobristHasherNumpy:
    def __init__(self, planes: int, rows: int, cols: int):
        self.planes = planes
        self.rows = rows
        self.cols = cols
        self.size = planes * rows * cols
        self.keys: np.ndarray = self._create_zobrist_keys()

    def _create_zobrist_keys(self) -> np.ndarray:
        """Create random zobrist keys."""
        return np.random.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(self.planes, self.rows, self.cols),
            dtype=np.int64,
        ).reshape(self.size)

    def zobrist_hash_boards(self, boards: np.ndarray) -> list[int]:
        assert boards.shape[1:] == (self.planes, self.rows, self.cols), f'Invalid shape: {boards.shape}'

        boards = boards.astype(np.int64)
        boards_flat = boards.reshape(boards.shape[0], self.size)
        selected_keys = boards_flat * self.keys
        return np.bitwise_xor.reduce(selected_keys, axis=1).tolist()

    def zobrist_hash_board(self, board: np.ndarray) -> int:
        assert board.shape == (self.planes, self.rows, self.cols), f'Invalid shape: {board.shape}'

        board = board.astype(np.int64)
        board_flat = board.reshape(self.size)
        selected_keys = board_flat * self.keys
        return int(np.bitwise_xor.reduce(selected_keys))
