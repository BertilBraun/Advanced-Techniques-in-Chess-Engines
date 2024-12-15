from __future__ import annotations

import torch
import numpy as np
from typing import List

from AIZeroConnect4Bot.src.games.Game import Game
from AIZeroConnect4Bot.src.games.checkers.CheckersBoard import CheckersBoard, CheckersMove

ROW_COUNT = 8
COLUMN_COUNT = 8
ENCODING_CHANNELS = 4


ZOBRIST_TABLES: dict[torch.device, torch.Tensor] = {}


class CheckersGame(Game[CheckersMove]):
    @property
    def null_move(self) -> CheckersMove:
        return -1, -1

    @property
    def action_size(self) -> int:
        # TODO optimize this, Tons of moves are invalid and should not be considered
        board_dimension = ROW_COUNT * COLUMN_COUNT
        return board_dimension * board_dimension

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

    @property
    def network_properties(self) -> tuple[int, int]:
        NUM_RES_BLOCKS = 10
        NUM_HIDDEN = 128
        return NUM_RES_BLOCKS, NUM_HIDDEN

    @property
    def average_num_moves_per_game(self) -> int:
        return 30

    def get_canonical_board(self, board: CheckersBoard) -> np.ndarray:
        # TODO verify correctness once valid move generation and make move are correctly implemented
        # turn the 4 bitboards into a single 4x8x8 tensor
        # board.black_kings, board.black_pieces, board.white_kings, board.white_pieces
        # The bitboards are 64 bit integers, so we need to convert them to a 8x8 tensor first
        # Then we can stack them together to get a 4x8x8 tensor
        def bitfield_to_tensor(bitfield: np.uint64, flipped: bool) -> np.ndarray:
            # turn 64 bit integer into a list of 8x 8bit integers, then use np.unpackbits to get a 8x8 tensor

            tensor = np.unpackbits(np.frombuffer(bitfield.tobytes(), dtype=np.uint8)).reshape(ROW_COUNT, COLUMN_COUNT)
            if flipped:
                tensor = tensor[::-1]
            return tensor

        if board.current_player == 1:
            return np.stack(
                [
                    bitfield_to_tensor(board.black_kings, flipped=False),
                    bitfield_to_tensor(board.black_pieces, flipped=False),
                    bitfield_to_tensor(board.white_kings, flipped=False),
                    bitfield_to_tensor(board.white_pieces, flipped=False),
                ]
            )
        else:
            return np.stack(
                [
                    bitfield_to_tensor(board.white_kings, flipped=True),
                    bitfield_to_tensor(board.white_pieces, flipped=True),
                    bitfield_to_tensor(board.black_kings, flipped=True),
                    bitfield_to_tensor(board.black_pieces, flipped=True),
                ]
            )

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'

        # each board is a 4x8x8 tensor
        # each entry is either 0 or 1
        # We multiply each of the 4 channels with a different 8x8 zobrist key
        # Then sum them up to get a single hash for each board
        # TODO no idea how fast this is, might be slow, but should be fast and correct enough for now

        if boards.device not in ZOBRIST_TABLES:
            zobrist_table = torch.randint(
                low=-(2**63),
                high=2**63 - 1,
                size=(ENCODING_CHANNELS, ROW_COUNT * COLUMN_COUNT),
                dtype=torch.long,
                device=boards.device,
            )
            ZOBRIST_TABLES[boards.device] = zobrist_table
        else:
            zobrist_table = ZOBRIST_TABLES[boards.device]

        boards = boards.reshape(-1, ENCODING_CHANNELS, ROW_COUNT * COLUMN_COUNT)
        for i in range(ENCODING_CHANNELS):
            boards[:, i] *= zobrist_table[i]

        return torch.sum(boards, dim=(1, 2)).tolist()

    def encode_move(self, move: CheckersMove) -> int:
        move_from, move_to = move
        board_dimension = ROW_COUNT * COLUMN_COUNT

        assert 0 <= move_from < board_dimension, f'Invalid move: {move}'
        assert 0 <= move_to < board_dimension, f'Invalid move: {move}'

        return move_from * board_dimension + move_to

    def decode_move(self, move: int) -> CheckersMove:
        board_dimension = ROW_COUNT * COLUMN_COUNT
        move_from = move // board_dimension
        move_to = move % board_dimension

        assert 0 <= move_from < board_dimension, f'Invalid move: {move}'
        assert 0 <= move_to < board_dimension, f'Invalid move: {move}'

        return move_from, move_to

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        return [
            # Original board
            (board, action_probabilities),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            (np.flip(board, axis=2), np.flip(action_probabilities)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # TODO possibly doable? If also the action probabilities are flipped?
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> CheckersBoard:
        return CheckersBoard()


if __name__ == '__main__':

    def print_board_side_by_side(board1, board2, c):
        print('Board 1', ' ' * 8, 'Board 2')
        for row1, row2 in zip(board1, board2):
            for c1 in row1:
                print('.' if not c1 else c, end=' ')
            print(' | ', end='')
            for c2 in row2:
                print('.' if not c2 else c, end=' ')
            print()

    def compare_boards(board1, board2):
        print_board_side_by_side(board1[1], board2[1], 'X')
        print_board_side_by_side(board1[3], board2[3], 'O')

    g = CheckersGame()
    b = g.get_initial_board()
    b1 = g.get_canonical_board(b)
    # print(g.get_canonical_board(b))
    print(b.get_valid_moves())
    # print('Hash:', g.hash_boards(torch.tensor(np.array([g.get_canonical_board(b)]))))
    # print('Hash:', g.hash_boards(torch.tensor(np.array([g.get_canonical_board(b)]))))
    b.make_move(b.get_valid_moves()[0])
    b2 = g.get_canonical_board(b)
    print(b.get_valid_moves())
    # print('After move')
    # print('Hash:', g.hash_boards(torch.tensor(np.array([g.get_canonical_board(b)]))))
    # print('After move')
    b.make_move(b.get_valid_moves()[0])
    # print('After move')
    # print('Hash:', g.hash_boards(torch.tensor(np.array([g.get_canonical_board(b)]))))
    # print(g.get_canonical_board(b))
    b3 = g.get_canonical_board(b)
    # compare_boards(b1, b2)
    # compare_boards(b2, b3)
    # compare_boards(b1, b3)
