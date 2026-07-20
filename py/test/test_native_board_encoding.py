from __future__ import annotations

import chess
import pytest

AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

if not hasattr(AlphaZeroCpp, 'encode_board_compressed'):
    pytest.skip('AlphaZeroCpp must be rebuilt before native encoding tests run.', allow_module_level=True)

ALL_SQUARES = (1 << 64) - 1


def expected_compressed_encoding(fen: str) -> tuple[list[int], list[int]]:
    board = chess.Board(fen)
    canonical = board if board.turn == chess.WHITE else board.mirror()

    binary_planes = [
        int(canonical.pieces(piece_type, color))
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in chess.PIECE_TYPES
    ]
    binary_planes.extend(
        (
            ALL_SQUARES * canonical.has_kingside_castling_rights(chess.WHITE),
            ALL_SQUARES * canonical.has_queenside_castling_rights(chess.WHITE),
            ALL_SQUARES * canonical.has_kingside_castling_rights(chess.BLACK),
            ALL_SQUARES * canonical.has_queenside_castling_rights(chess.BLACK),
            int(canonical.occupied_co[chess.WHITE]),
            int(canonical.occupied_co[chess.BLACK]),
            int(canonical.checkers()),
            0 if canonical.ep_square is None else int(chess.BB_SQUARES[canonical.ep_square]),
            0,
            0,
        )
    )

    scalar_planes = [
        len(canonical.pieces(piece_type, chess.WHITE)) - len(canonical.pieces(piece_type, chess.BLACK))
        for piece_type in chess.PIECE_TYPES
    ]
    scalar_planes.append(min(canonical.halfmove_clock, 100))
    return binary_planes, scalar_planes


@pytest.mark.parametrize(
    'fen',
    (
        'r3k2r/pppp3p/8/4pP2/8/8/PPPP1PPP/R3K2R w Qk e6 17 9',
        'r3k2r/pppp1ppp/8/8/4Pp2/8/PPPP3P/R3K2R b Kq e3 17 9',
        '4k3/8/8/8/8/8/4R3/4K3 b - - 142 1',
    ),
)
def test_native_encoding_matches_canonical_plane_semantics(fen: str) -> None:
    encoded_binary, encoded_scalar = AlphaZeroCpp.encode_board_compressed(fen)

    assert (encoded_binary, encoded_scalar) == expected_compressed_encoding(fen)
