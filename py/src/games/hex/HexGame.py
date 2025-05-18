from __future__ import annotations

import numpy as np
from typing import List, Tuple

from src.games.Game import Game

from src.games.hex.HexBoard import HexBoard, SIZE

HexMove = int  # A Hex move is a single position on the board (flattened index)


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
        2 – empty                       (1s)
        3 - current player (1s)
        """
        # return np.stack(
        #     [
        #         (board.board == 1),
        #         (board.board == -1),
        #         (board.board == 0),
        #         np.full_like(board.board, board.current_player == 1),
        #     ],
        #     axis=0,
        # ).astype(np.float32)

        # Put the side to move’s stones in +1
        pos = board.board * board.current_player  # +1 == me, –1 == them

        # If the side to move is the LR-player, rotate 90° so *I* now aim N-S
        if board.current_player == -1:
            pos = np.rot90(pos, k=1)  # counter-clockwise

        planes = np.stack(
            [(pos == 1), (pos == -1), (pos == 0)],
            dtype=np.float32,
        )
        return planes.astype(np.float32)

    def encode_move(self, move: HexMove) -> int:
        assert 0 <= move < SIZE * SIZE, f'Invalid move: {move}'
        return move

    def decode_move(self, move_idx: int) -> HexMove:
        assert 0 <= move_idx < SIZE * SIZE, f'Invalid move index: {move_idx}'
        return move_idx

    def symmetric_variations(
        self, board: np.ndarray, visit_counts: List[Tuple[int, int]]
    ) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """
        Return the 4 symmetries of an NxN hex board: identity, 180° rotation, flip top↔bottom, flip left↔right.
        Each symmetry is a tuple of (board, visit_counts).
        """

        def remap_counts(map_fn):
            out: list[tuple[int, int]] = []
            for idx, n in visit_counts:
                r, c = divmod(self.decode_move(idx), SIZE)
                nr, nc = map_fn(r, c)
                idx2 = self.encode_move(nr * SIZE + nc)
                out.append((idx2, n))
            return out

        syms: List[Tuple[np.ndarray, List[Tuple[int, int]]]] = []
        # 0) identity
        syms.append((board, visit_counts))

        # 1) 180° rotation (preserves N-S orientation)
        syms.append(
            (
                np.rot90(board, k=2, axes=(1, 2)),
                remap_counts(lambda r, c: (SIZE - 1 - r, SIZE - 1 - c)),
            )
        )

        # 2) flip top↔bottom
        syms.append(
            (
                np.flip(board, axis=1),
                remap_counts(lambda r, c: (SIZE - 1 - r, c)),
            )
        )

        # 3) flip left↔right
        syms.append(
            (
                np.flip(board, axis=2),
                remap_counts(lambda r, c: (r, SIZE - 1 - c)),
            )
        )

        return syms


if __name__ == '__main__':
    game = HexGame()
    board = game.get_initial_board()
    for _ in range(10):
        move = np.random.choice(board.get_valid_moves())
        board.make_move(move)

    print(board)
    print(board.current_player)

    print(game.get_canonical_board(board))
    move = np.random.choice(board.get_valid_moves())
    r, c = divmod(move, SIZE)
    print(move, r, c)
    for bv, mv in game.symmetric_variations(game.get_canonical_board(board), [(move, 5)]):
        b = game.get_initial_board()
        b.board = bv[0] - bv[1]
        r, c = divmod(mv[0][0], SIZE)
        b.board[r, c] = mv[0][1]
        print(b)
