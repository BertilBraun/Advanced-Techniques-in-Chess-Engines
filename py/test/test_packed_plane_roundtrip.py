from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.representation import (
    PackedPlanePayload,
    decode_packed_planes,
    encode_packed_planes,
)

CHESS_LAYOUT = CHESS_STATE_CONTRACT.packed_plane_layout
BINARY_CHANNELS = CHESS_STATE_CONTRACT.representation.binary_channels
SCALAR_CHANNELS = CHESS_STATE_CONTRACT.representation.scalar_channels
FIXTURES = json.loads(
    (Path(__file__).parent / 'fixtures' / 'chess_packed_plane_fixtures.json').read_text(encoding='utf-8')
)


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda fixture: fixture['fen'])
def test_python_codec_round_trips_the_native_fixture_payloads(fixture: dict[str, str]) -> None:
    payload = PackedPlanePayload(bytes.fromhex(fixture['packed_hex']))
    decoded = decode_packed_planes(payload, CHESS_LAYOUT, BINARY_CHANNELS, SCALAR_CHANNELS)
    assert decoded.shape == (CHESS_STATE_CONTRACT.representation.channels, 8, 8)
    re_encoded = encode_packed_planes(decoded, CHESS_LAYOUT, BINARY_CHANNELS, SCALAR_CHANNELS)
    assert bytes(re_encoded) == bytes(payload)
