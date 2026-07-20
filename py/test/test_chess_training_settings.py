from src.games.chess.ChessSettings import training


def test_chess_training_uses_large_optimizer_batch() -> None:
    assert training.global_batch_size == 1024
    assert training.local_batch_size == 1024
