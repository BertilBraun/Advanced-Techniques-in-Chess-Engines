"""Builds and verifies the permutation between Lc0's 1858 policy indices and this project's 1880.

The mapping is derived empirically rather than assumed: positions are walked, and for every legal
move the project's own action id is compared against Lc0's index for the same move in the canonical
(side-to-move) frame. A move that ever maps two different ways is a hard failure, because the whole
teacher experiment rests on this table being exact.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.games.chess.contract import CHESS_ACTION_SIZE
from src.util.atomic_file import write_text_atomically

LC0_POLICY_SIZE = 1858
UCI_PATTERN = re.compile(r'"([a-h][1-8][a-h][1-8][qrbn]?)"')


@dataclass(frozen=True)
class PolicyMap:
    lc0_index_to_action_id: tuple[int, ...]
    covered_action_ids: tuple[int, ...]
    positions_walked: int


def parse_lc0_move_table(source: Path) -> tuple[str, ...]:
    """Extracts kIdxToMove from Lc0's bitboard.cc in declaration order."""
    moves = UCI_PATTERN.findall(source.read_text(encoding='utf-8'))
    if len(moves) < LC0_POLICY_SIZE:
        raise SystemExit(f'Found only {len(moves)} UCI strings in {source}; expected {LC0_POLICY_SIZE}.')
    table = tuple(moves[:LC0_POLICY_SIZE])
    if len(set(table)) != LC0_POLICY_SIZE:
        raise SystemExit('Lc0 move table contains duplicates; the parse picked up unrelated strings.')
    return table


def flip_uci_ranks(move_uci: str) -> str:
    """Mirrors a move into the side-to-move frame, which is how both engines index policy."""
    flipped = ''.join(str(9 - int(character)) if character.isdigit() else character for character in move_uci[:4])
    return flipped + move_uci[4:]


def build(move_table: tuple[str, ...], position_count: int, seed: int) -> PolicyMap:
    import AlphaZeroCpp

    lc0_index_of = {move: index for index, move in enumerate(move_table)}
    mapping: dict[int, int] = {}
    generator = random.Random(seed)
    positions_walked = 0

    for _ in range(position_count):
        position = AlphaZeroCpp.ChessPosition()
        for _ in range(generator.randint(0, 160)):
            if position.is_terminal:
                break
            positions_walked += 1
            legal_action_ids = position.legal_actions()
            for action_id in legal_action_ids:
                move_uci = position.action_uci(action_id)
                canonical = move_uci if position.current_player == 1 else flip_uci_ranks(move_uci)
                lc0_index = lc0_index_of.get(canonical)
                if lc0_index is None:
                    raise SystemExit(f'Legal move {move_uci} (canonical {canonical}) is absent from the Lc0 table.')
                previous = mapping.get(lc0_index)
                if previous is not None and previous != action_id:
                    raise SystemExit(
                        f'Lc0 index {lc0_index} ({canonical}) maps to both action {previous} and {action_id}.'
                    )
                mapping[lc0_index] = action_id
            position = position.child(generator.choice(legal_action_ids))

    unmapped = LC0_POLICY_SIZE - len(mapping)
    table = tuple(mapping.get(index, -1) for index in range(LC0_POLICY_SIZE))
    print(f'Walked {positions_walked} positions; {len(mapping)} of {LC0_POLICY_SIZE} Lc0 indices covered.')
    if unmapped:
        print(f'{unmapped} Lc0 indices were never reached and are written as -1.')
    return PolicyMap(
        lc0_index_to_action_id=table,
        covered_action_ids=tuple(sorted(set(mapping.values()))),
        positions_walked=positions_walked,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lc0-bitboard-source', type=Path, required=True, help="Lc0's src/chess/bitboard.cc.")
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position-count', type=int, default=4000, help='Random games walked to cover the table.')
    parser.add_argument('--seed', type=int, default=20260926)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    move_table = parse_lc0_move_table(arguments.lc0_bitboard_source)
    policy_map = build(move_table, arguments.position_count, arguments.seed)
    payload = {
        'lc0_policy_size': LC0_POLICY_SIZE,
        'project_action_size': CHESS_ACTION_SIZE,
        'positions_walked': policy_map.positions_walked,
        'covered_lc0_indices': sum(1 for entry in policy_map.lc0_index_to_action_id if entry >= 0),
        'covered_action_ids': len(policy_map.covered_action_ids),
        'lc0_index_to_action_id': list(policy_map.lc0_index_to_action_id),
    }
    write_text_atomically(arguments.output, json.dumps(payload, indent=2) + '\n')
    print(f'Wrote {arguments.output}')


if __name__ == '__main__':
    main()
