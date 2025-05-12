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
        canonical = board.board * board.current_player
        planes = np.stack([(canonical == 1), (canonical == -1), (canonical == 0)])
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
        Return the 4 symmetries of an NxN hex board: identity, 180° rotation,
        main-diagonal reflection (transpose), and anti-diagonal reflection.
        """
        syms: List[Tuple[np.ndarray, List[Tuple[int, int]]]] = []
        # 1) Identity
        syms.append((board, visit_counts))
        # 2) 180° rotation
        b_rot = np.rot90(board, k=2, axes=(1, 2))
        vc_rot: List[Tuple[int, int]] = []
        for idx, count in visit_counts:
            r, c = divmod(self.decode_move(idx), SIZE)
            r2, c2 = SIZE - 1 - r, SIZE - 1 - c
            idx2 = self.encode_move(r2 * SIZE + c2)
            vc_rot.append((idx2, count))
        syms.append((b_rot, vc_rot))
        # 3) Main-diagonal reflection (transpose)
        b_t = board.transpose((0, 2, 1))
        vc_t: List[Tuple[int, int]] = []
        for idx, count in visit_counts:
            r, c = divmod(self.decode_move(idx), SIZE)
            idx2 = self.encode_move(c * SIZE + r)
            vc_t.append((idx2, count))
        syms.append((b_t, vc_t))
        # 4) Anti-diagonal reflection
        # r2 = SIZE-1-c, c2 = SIZE-1-r
        b_ad = np.rot90(b_t, k=2, axes=(1, 2))  # rotate transposed by 180 is anti-diagonal
        vc_ad: List[Tuple[int, int]] = []
        for idx, count in visit_counts:
            r, c = divmod(self.decode_move(idx), SIZE)
            r2, c2 = SIZE - 1 - c, SIZE - 1 - r
            idx2 = self.encode_move(r2 * SIZE + c2)
            vc_ad.append((idx2, count))
        syms.append((b_ad, vc_ad))
        return syms
