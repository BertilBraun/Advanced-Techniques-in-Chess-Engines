import random
from src.games.chess.ChessGame import ChessGame


def test_move_mapping_count():
    game = ChessGame()
    assert len(game.move2index) == 1814  # 1880 with all promotion peaces, 1968 with black promotions


def fuzz_test_move_encoding_and_decoding():
    game = ChessGame()
    for _ in range(100):
        b = game.get_initial_board()
        while not b.is_game_over():
            for move in b.get_valid_moves():
                assert game.decode_move(game.encode_move(move, b), b) == move

            b.make_move(random.choice(b.get_valid_moves()))


def display_sample_boards():
    game = ChessGame()
    b = game.get_initial_board()

    for _ in range(10):
        print(b)
        print(f'Current player: {b.current_player}')
        for layer_name, layer in zip(
            [
                'white pawns',
                'white knights',
                'white bishops',
                'white rooks',
                'white queens',
                'white kings',
                'black pawns',
                'black knights',
                'black bishops',
                'black rooks',
                'black queens',
                'black kings',
                'castling rights',
                'castling rights',
                'castling rights',
                'castling rights',
                'en passant',
            ],
            game.get_canonical_board(b),
        ):
            print(f'{layer_name}:')
            print(layer)
            print()
        print('-' * 20)
        move = random.choice(b.get_valid_moves())
        print(f'Move: {move}')
        b.make_move(move)


if __name__ == '__main__':
    test_move_mapping_count()
    fuzz_test_move_encoding_and_decoding()
    display_sample_boards()
    print('All tests passed!')
