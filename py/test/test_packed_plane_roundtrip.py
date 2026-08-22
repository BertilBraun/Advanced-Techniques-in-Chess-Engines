from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.games.representation import (
    PackedPlaneLayout,
    PackedPlanePayload,
    decode_packed_planes,
    encode_packed_planes,
)

CHESS_LAYOUT = PackedPlaneLayout(board_size=8, binary_plane_count=22, scalar_count=7)
BINARY_CHANNELS = tuple(range(22))
SCALAR_CHANNELS = tuple(range(22, 29))
FIXTURES = json.loads(
    (Path(__file__).parent / 'fixtures' / 'chess_packed_plane_fixtures.json').read_text(encoding='utf-8')
)


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda fixture: fixture['fen'])
def test_python_codec_round_trips_the_native_fixture_payloads(fixture: dict[str, str]) -> None:
    payload = PackedPlanePayload(bytes.fromhex(fixture['packed_hex']))
    decoded = decode_packed_planes(payload, CHESS_LAYOUT, BINARY_CHANNELS, SCALAR_CHANNELS)
    assert decoded.shape == (29, 8, 8)
    re_encoded = encode_packed_planes(decoded, CHESS_LAYOUT, BINARY_CHANNELS, SCALAR_CHANNELS)
    assert bytes(re_encoded) == bytes(payload)
