"""Regenerates the packed-plane codec fixtures against the current native encoder.

The committed fixtures pin real payloads so the Python codec cannot drift from the C++ one. They are
layout-specific, so changing the encoding invalidates them and they must be re-recorded from the
native extension rather than edited by hand.

Positions are built by replaying moves, not by loading a FEN, so the history planes carry real
content and the round trip exercises the whole payload.

Requires the native extension, so it runs on the node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.util.atomic_file import write_text_atomically

# Chosen for payload variety: castling rights alive, an en-passant square, promotions, a repetition,
# and a long reversible run that saturates the rule-50 scalar.
FIXTURE_LINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('opening-castling-rights', ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6')),
    ('en-passant-available', ('e2e4', 'a7a6', 'e4e5', 'd7d5')),
    ('castled-both-sides', ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'e1g1', 'e8g8')),
    ('knight-shuffle-repetition', ('g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8')),
    ('queenside-development', ('d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3', 'g8f6', 'c1g5', 'f8e7')),
)


def build_fixture(name: str, moves_uci: tuple[str, ...]) -> dict[str, str]:
    position = CHESS_STATE_CONTRACT.initial_position()
    for move_uci in moves_uci:
        position = CHESS_STATE_CONTRACT.child_position(position, position.action_id_from_uci(move_uci))
    payload = CHESS_STATE_CONTRACT.encode_network_input(position)
    return {
        'name': name,
        'fen': position.fen,
        'moves': ' '.join(moves_uci),
        'packed_hex': bytes(payload.payload).hex(),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('test/fixtures/chess_packed_plane_fixtures.json'),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    fixtures = [build_fixture(name, moves) for name, moves in FIXTURE_LINES]
    payload_bytes = {len(bytes.fromhex(fixture['packed_hex'])) for fixture in fixtures}
    expected = CHESS_STATE_CONTRACT.packed_plane_layout.payload_bytes
    if payload_bytes != {expected}:
        raise SystemExit(f'Native encoder produced payloads of {payload_bytes} bytes, expected {expected}.')
    write_text_atomically(arguments.output, json.dumps(fixtures, indent=2) + '\n')
    print(f'Wrote {len(fixtures)} fixtures of {expected} bytes each to {arguments.output}')


if __name__ == '__main__':
    main()
