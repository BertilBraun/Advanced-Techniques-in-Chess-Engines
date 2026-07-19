import chess
import pytest

from src.games.chess.ChessGame import ChessGame, normalize_move_for_action_space


@pytest.mark.parametrize('promotion', (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT))
def test_all_promotions_have_distinct_round_trip_actions(promotion: chess.PieceType) -> None:
    board = ChessGame().get_initial_board()
    board.set_fen('8/3P4/8/8/8/8/8/k6K w - - 0 1')
    game = ChessGame()
    move = chess.Move(chess.D7, chess.D8, promotion=promotion)

    normalized = normalize_move_for_action_space(move, board)
    encoded = game.encode_move(normalized, board)

    assert normalized == move
    assert game.decode_move(encoded, board) == move


def test_chess_action_space_includes_underpromotions() -> None:
    assert ChessGame().action_size == 1880
