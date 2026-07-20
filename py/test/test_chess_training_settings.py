from src.games.chess.ChessSettings import training


def test_chess_training_uses_large_optimizer_batch() -> None:
    assert training.batch_size == 1024
