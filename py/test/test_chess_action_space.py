import chess

from src.games.chess.ChessGame import ChessGame, normalize_move_for_action_space


def test_underpromotion_is_normalized_to_supported_queen_promotion() -> None:
    board = ChessGame().get_initial_board()
    board.set_fen('8/3P4/8/8/8/8/8/k6K w - - 0 1')
    underpromotion = chess.Move.from_uci('d7d8r')

    normalized = normalize_move_for_action_space(underpromotion, board)

    assert normalized == chess.Move.from_uci('d7d8q')
