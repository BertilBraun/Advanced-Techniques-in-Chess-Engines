import random
from src.games.ChessGame import ChessGame


def test_move_mapping_count():
    game = ChessGame()
    assert len(game.move2index) == 1968


def fuzz_test_move_encoding_and_decoding():
    game = ChessGame()
    for _ in range(100):
        b = game.get_initial_board()
        while not b.is_game_over():
            for move in b.get_valid_moves():
                assert game.decode_move(game.encode_move(move)) == move

            b.make_move(random.choice(b.get_valid_moves()))


if __name__ == '__main__':
    test_move_mapping_count()
    fuzz_test_move_encoding_and_decoding()
    print('All tests passed!')
