from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
from AlphaZeroCpp import InferenceClientParams, MCTS, MCTSNode, MCTSParams, new_root, new_root_with_history


KNIGHT_CYCLE = (
    'g1f3',
    'g8f6',
    'f3g1',
    'f6g8',
)


@dataclass(frozen=True)
class Arguments:
    model: Path
    device: int


def repetition_child(root: MCTSNode) -> MCTSNode:
    matching_children = [child for child in root.children if child.move == 'f6g8']
    if len(matching_children) != 1:
        raise ValueError(f'Expected one f6g8 child, found {len(matching_children)}.')
    return matching_children[0]


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--device', type=int, default=3)
    namespace = parser.parse_args()
    return Arguments(model=namespace.model, device=namespace.device)


def main() -> None:
    args = parse_arguments()
    moves = KNIGHT_CYCLE * 2
    board = chess.Board()
    for move in moves:
        board.push_uci(move)

    history_blind_root = new_root(board.fen())
    history_aware_root = new_root_with_history(chess.STARTING_FEN, moves)

    assert not history_blind_root.is_terminal
    assert history_aware_root.is_terminal
    print('history_blind_terminal=False')
    print('history_aware_terminal=True')

    moves_before_third_occurrence = KNIGHT_CYCLE + KNIGHT_CYCLE[:-1]
    board_before_third_occurrence = chess.Board()
    for move in moves_before_third_occurrence:
        board_before_third_occurrence.push_uci(move)

    blind_parent = new_root(board_before_third_occurrence.fen())
    aware_parent = new_root_with_history(chess.STARTING_FEN, moves_before_third_occurrence)
    mcts = MCTS(
        InferenceClientParams(args.device, str(args.model), 16, 500, 10_000),
        MCTSParams(1, 8, 8, 1.0, 0.3, 0.0, 0, 1),
    )
    results = mcts.search([(blind_parent, False), (aware_parent, False)])
    blind_child = repetition_child(results.results[0].root)
    aware_child = repetition_child(results.results[1].root)

    assert not blind_child.is_terminal
    assert aware_child.is_terminal
    print('history_blind_child_terminal=False')
    print('history_aware_child_terminal=True')

    castling_moves = (
        'e2e4',
        'e7e5',
        'g1f3',
        'b8c6',
        'f1c4',
        'g8f6',
        'e1g1',
    )
    castled = chess.Board()
    for move in castling_moves:
        castled.push_uci(move)
    castled_root = new_root_with_history(chess.STARTING_FEN, castling_moves)
    assert castled_root.fen == castled.fen()
    print('standard_uci_castling_replay=True')


if __name__ == '__main__':
    main()
