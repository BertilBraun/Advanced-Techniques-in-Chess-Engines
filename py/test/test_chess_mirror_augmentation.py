import numpy as np
import pytest

from src.games.chess.contract import (
    CHESS_STATE_CONTRACT,
    OPPONENT_KING_SIDE_CASTLING_PLANE,
    OPPONENT_QUEEN_SIDE_CASTLING_PLANE,
    OWN_KING_SIDE_CASTLING_PLANE,
    OWN_QUEEN_SIDE_CASTLING_PLANE,
)
from src.games.representation import PackedPlanePayload, decode_packed_planes, encode_packed_planes

REPRESENTATION = CHESS_STATE_CONTRACT.representation
CASTLING_PLANES = (
    OWN_KING_SIDE_CASTLING_PLANE,
    OWN_QUEEN_SIDE_CASTLING_PLANE,
    OPPONENT_KING_SIDE_CASTLING_PLANE,
    OPPONENT_QUEEN_SIDE_CASTLING_PLANE,
)


def _decode(payload: PackedPlanePayload) -> np.ndarray:
    return decode_packed_planes(
        payload,
        REPRESENTATION.packed_planes,
        REPRESENTATION.binary_channels,
        REPRESENTATION.scalar_channels,
    )


def _synthetic_state() -> np.ndarray:
    generator = np.random.default_rng(3)
    state = np.zeros((29, 8, 8), dtype=np.int8)
    for channel in REPRESENTATION.binary_channels:
        state[channel] = (generator.random((8, 8)) < 0.2).astype(np.int8)
    state[OWN_KING_SIDE_CASTLING_PLANE] = 1
    state[OWN_QUEEN_SIDE_CASTLING_PLANE] = 0
    state[OPPONENT_KING_SIDE_CASTLING_PLANE] = 1
    state[OPPONENT_QUEEN_SIDE_CASTLING_PLANE] = 0
    for scalar_index, channel in enumerate(REPRESENTATION.scalar_channels):
        state[channel] = scalar_index - 3
    return state


def test_file_mirror_flips_files_and_swaps_castling_planes() -> None:
    state = _synthetic_state()
    payload = encode_packed_planes(
        state,
        REPRESENTATION.packed_planes,
        REPRESENTATION.binary_channels,
        REPRESENTATION.scalar_channels,
    )

    mirrored = _decode(CHESS_STATE_CONTRACT.transform_encoded_state(payload, 1))

    expected = np.flip(state, axis=2).copy()
    expected[list(CASTLING_PLANES)] = expected[
        [
            OWN_QUEEN_SIDE_CASTLING_PLANE,
            OWN_KING_SIDE_CASTLING_PLANE,
            OPPONENT_QUEEN_SIDE_CASTLING_PLANE,
            OPPONENT_KING_SIDE_CASTLING_PLANE,
        ]
    ]
    assert np.array_equal(mirrored, expected)
    assert np.all(mirrored[OWN_KING_SIDE_CASTLING_PLANE] == 0)
    assert np.all(mirrored[OWN_QUEEN_SIDE_CASTLING_PLANE] == 1)
    assert np.all(mirrored[OPPONENT_KING_SIDE_CASTLING_PLANE] == 0)
    assert np.all(mirrored[OPPONENT_QUEEN_SIDE_CASTLING_PLANE] == 1)


def test_identity_augmentation_returns_the_original_payload() -> None:
    state = _synthetic_state()
    payload = encode_packed_planes(
        state,
        REPRESENTATION.packed_planes,
        REPRESENTATION.binary_channels,
        REPRESENTATION.scalar_channels,
    )

    assert CHESS_STATE_CONTRACT.transform_encoded_state(payload, 0) == payload


def _file_mirrored_board(board: str) -> str:
    return '/'.join(rank[::-1] for rank in board.split('/'))


def _file_mirrored_square(square: str) -> str:
    return chr(ord('a') + 7 - (ord(square[0]) - ord('a'))) + square[1]


def _file_mirrored_fen_without_castling(fen: str) -> str:
    board, side, _, en_passant, halfmove, fullmove = fen.split()
    mirrored_en_passant = '-' if en_passant == '-' else _file_mirrored_square(en_passant)
    return f'{_file_mirrored_board(board)} {side} - {mirrored_en_passant} {halfmove} {fullmove}'


def _swapped_castling_fen(fen: str) -> str:
    board, side, castling, en_passant, halfmove, fullmove = fen.split()
    assert castling != '-'
    swapped = {'K': 'Q', 'Q': 'K', 'k': 'q', 'q': 'k'}
    rights = ''.join(sorted((swapped[right] for right in castling), key='KQkq'.index))
    return f'{board} {side} {rights} {en_passant} {halfmove} {fullmove}'


@pytest.mark.parametrize(
    'fen',
    (
        'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1',
        'r3k2r/pppqppbp/2np1np1/8/2PP4/2N2NP1/PP2PPBP/R2QK2R b KQkq - 0 1',
        'rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3',
    ),
)
def test_mirrored_tensor_matches_native_encoding_for_positions_with_castling_rights(fen: str) -> None:
    native = pytest.importorskip('AlphaZeroCpp')

    payload = CHESS_STATE_CONTRACT.encode_network_input(native.ChessPosition(fen))
    mirrored = _decode(CHESS_STATE_CONTRACT.transform_encoded_state(payload, 1))

    # Standard FEN cannot express the mirrored castling rights, so the expected tensor combines the
    # native encoding of the mirrored board with the constant castling planes of the swapped-rights position.
    board_expectation = _decode(
        CHESS_STATE_CONTRACT.encode_network_input(native.ChessPosition(_file_mirrored_fen_without_castling(fen)))
    )
    rights_expectation = _decode(
        CHESS_STATE_CONTRACT.encode_network_input(native.ChessPosition(_swapped_castling_fen(fen)))
    )
    expected = board_expectation.copy()
    expected[list(CASTLING_PLANES)] = rights_expectation[list(CASTLING_PLANES)]

    assert np.array_equal(mirrored, expected)
