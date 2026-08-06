from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.Encoding import decode_board_state, encode_board_state
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.contract import CHESS_STATE_CONTRACT


AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp, 'encode_board_packed_bytes'):
    pytest.skip('AlphaZeroCpp must be rebuilt before native packed-plane tests run.', allow_module_level=True)


_FIXTURE_PATH = Path(__file__).with_name('fixtures') / 'chess_packed_plane_fixtures.json'
_FIXTURES: tuple[dict[str, str], ...] = tuple(json.loads(_FIXTURE_PATH.read_text(encoding='utf-8')))


@pytest.mark.parametrize('fixture', _FIXTURES)
def test_native_and_python_packed_layout_match_shared_fixture(fixture: dict[str, str]) -> None:
    fen = fixture['fen']
    expected_payload = bytes.fromhex(fixture['packed_hex'])
    board = ChessBoard.from_fen(fen)
    canonical_state = CHESS_STATE_CONTRACT.canonical_board(board).astype(np.int8, copy=False)

    python_encoded = encode_board_state(canonical_state)
    native_payload = AlphaZeroCpp.encode_board_packed_bytes(fen)
    native_binary, native_scalar = AlphaZeroCpp.encode_board_compressed(fen)

    assert python_encoded.payload == expected_payload
    assert native_payload == expected_payload
    assert tuple(native_binary) == tuple(
        int.from_bytes(expected_payload[index * 8 : (index + 1) * 8], 'little') for index in range(len(native_binary))
    )
    assert tuple(native_scalar) == tuple(int(value) for value in expected_payload[len(native_binary) * 8 :])
    np.testing.assert_array_equal(decode_board_state(python_encoded), canonical_state)
