from __future__ import annotations

import numpy as np
from typing import List, Tuple

from src.games.Game import Game

from src.games.hex.HexBoard import HexBoard, SIZE

HexMove = int  # A Hex move is a single position on the board (flattened index)


def rotClockwise(r: int, c: int) -> tuple[int, int]:
    return (c, SIZE - 1 - r)


def rotCounterClockwise(r: int, c: int) -> tuple[int, int]:
    return (SIZE - 1 - c, r)


class HexGame(Game[HexMove]):
    @property
    def action_size(self) -> int:
        return SIZE * SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        # planes: current player, opponent, empty
        return (3, SIZE, SIZE)

    def get_initial_board(self) -> HexBoard:
        return HexBoard()

    def get_canonical_board(self, board: HexBoard) -> np.ndarray:
        """
        Planes:
        0 – stones of the side to move  (1s)
        1 – stones of the opponent      (1s)
        2 – empty cells                 (1s)
        """
        # Put the side to move’s stones in +1
        pos = board.board * board.current_player  # +1 == me, –1 == them

        # If the side to move is the LR-player, rotate 90° so *I* now aim N-S
        if board.current_player == -1:
            pos = np.rot90(pos, k=1)  # counter-clockwise

        return np.stack(
            [(pos == 1), (pos == -1), (pos == 0)],
            dtype=np.float32,
        )

    def encode_move(self, move: HexMove, board: HexBoard) -> int:
        assert 0 <= move < SIZE * SIZE, f'Invalid move: {move}'
        if board.current_player == 1:
            return move
        # If the side to move is the LR-player, rotate 90° so *I* now aim N-S
        r, c = divmod(move, SIZE)
        nr, nc = rotCounterClockwise(r, c)
        return nr * SIZE + nc

    def decode_move(self, move_idx: int, board: HexBoard) -> HexMove:
        assert 0 <= move_idx < SIZE * SIZE, f'Invalid move index: {move_idx}'
        if board.current_player == 1:
            return move_idx
        # If the side to move is the LR-player, rotate 90° so *I* now aim N-S
        r, c = divmod(move_idx, SIZE)
        nr, nc = rotClockwise(r, c)
        return nr * SIZE + nc

    def symmetric_variations(
        self, board: HexBoard, visit_counts: List[Tuple[int, int]]
    ) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """
        Return the 4 symmetries of an NxN hex board: identity, 180° rotation, flip top↔bottom, flip left↔right.
        Each symmetry is a tuple of (board, visit_counts).
        """

        def remap_counts(map_fn) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for idx, n in visit_counts:
                r, c = divmod(idx, SIZE)
                nr, nc = map_fn(r, c)
                idx2 = nr * SIZE + nc
                out.append((idx2, n))
            return out

        encoded_board = self.get_canonical_board(board)
        visit_counts = [(self.encode_move(idx, board), n) for idx, n in visit_counts]

        syms: List[Tuple[np.ndarray, List[Tuple[int, int]]]] = []
        # 0) identity
        syms.append((encoded_board, visit_counts))

        # 1) 180° rotation (preserves N-S orientation)
        syms.append(
            (
                np.rot90(encoded_board, k=2, axes=(1, 2)),
                remap_counts(lambda r, c: (SIZE - 1 - r, SIZE - 1 - c)),
            )
        )

        # 2) flip top↔bottom
        syms.append(
            (
                np.flip(encoded_board, axis=1),
                remap_counts(lambda r, c: (SIZE - 1 - r, c)),
            )
        )

        # 3) flip left↔right
        syms.append(
            (
                np.flip(encoded_board, axis=2),
                remap_counts(lambda r, c: (r, SIZE - 1 - c)),
            )
        )

        return syms


if __name__ == '__main__':
    game = HexGame()
    board = game.get_initial_board()
    for _ in range(10):
        for move in board.get_valid_moves():
            assert game.decode_move(game.encode_move(move, board), board) == move, f'Failed to decode move {move}'
        move = np.random.choice(board.get_valid_moves())
        board.make_move(move)

    for i in range(3):
        print(board)
        print(board.current_player)

        move = np.random.choice(board.get_valid_moves())
        r, c = divmod(move, SIZE)
        print(move, r, c)
        for bv, mv in game.symmetric_variations(board, [(move, 5)]):
            b = game.get_initial_board()
            b.board = bv[0] - bv[1]
            r, c = divmod(mv[0][0], SIZE)
            b.board[r, c] = mv[0][1]
            print(b)

        board.make_move(move)
