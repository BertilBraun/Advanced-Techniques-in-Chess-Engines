from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from AlphaZeroCpp import new_eval_root_with_history, new_root_with_history

STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
KNIGHT_CYCLE = (
    'g1f3',
    'g8f6',
    'f3g1',
    'f6g8',
)
EIGHT_PLY_CYCLE = (
    'g1f3',
    'g8f6',
    'b1c3',
    'b8c6',
    'f3g1',
    'f6g8',
    'c3b1',
    'c6b8',
)


class NativeRoot(Protocol):
    @property
    def is_terminal(self) -> bool: ...

    @property
    def repetition_count(self) -> int: ...


RootFactory = Callable[[str, tuple[str, ...]], NativeRoot]


def verify_root_factory(root_factory: RootFactory) -> None:
    second_occurrence = root_factory(STARTING_FEN, KNIGHT_CYCLE)
    third_occurrence = root_factory(STARTING_FEN, KNIGHT_CYCLE * 2)
    longer_cycle = root_factory(STARTING_FEN, EIGHT_PLY_CYCLE * 2)
    after_pawn_move = root_factory(STARTING_FEN, KNIGHT_CYCLE + ('e2e4',))

    if second_occurrence.repetition_count != 1 or second_occurrence.is_terminal:
        raise RuntimeError('Second occurrence was classified incorrectly.')
    if third_occurrence.repetition_count != 2 or not third_occurrence.is_terminal:
        raise RuntimeError('Third occurrence was classified incorrectly.')
    if longer_cycle.repetition_count != 2 or not longer_cycle.is_terminal:
        raise RuntimeError('Eight-ply repetition cycle was classified incorrectly.')
    if after_pawn_move.repetition_count != 0 or after_pawn_move.is_terminal:
        raise RuntimeError('Pawn move did not reset native repetition history.')


def main() -> None:
    verify_root_factory(new_root_with_history)
    verify_root_factory(new_eval_root_with_history)
    print(
        json.dumps(
            {
                'bounded_history': True,
                'irreversible_reset': True,
                'long_cycle': True,
                'threefold': True,
            },
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
