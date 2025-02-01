# test the flip and rotate visit counts. That should do the same as np.rot90(np.flip(encode_moves(moves), axis=2), k=k, axes=(1, 2))

import numpy as np

from src.games.tictactoe.TicTacToeGame import TicTacToeGame


if __name__ == '__main__':
    moves = [(i, i) for i in range(9)]
    board = np.array(
        [
            [list(range(9)) for _ in range(3)],
        ]
    ).reshape(3, 3, 3)

    game = TicTacToeGame()

    for bv, mv in game.symmetric_variations(board, moves):
        move_encoded = np.zeros(9, dtype=int)
        for move, count in mv:
            move_encoded[move] = count
        move_encoded = move_encoded.reshape(3, 3)

        print('All equal:', np.all(bv[0] == move_encoded))

    all_of_flipped = []

    for flip in [False, True]:
        for k in range(4):
            flipped = game._flip_and_rotate_visit_counts(moves, k, flip)
            flipped_arr = np.zeros(9, dtype=int)
            for move, count in flipped:
                flipped_arr[move] = count
            flipped_arr = flipped_arr.reshape(3, 3)
            all_of_flipped.append(flipped_arr)
            if flip:
                rot = np.rot90(np.flip(np.array([m for m, _ in moves]).reshape(3, 3), axis=1), k=k, axes=(0, 1))
            else:
                rot = np.rot90(np.array([m for m, _ in moves]).reshape(3, 3), k=k, axes=(0, 1))

            print(f'flip={flip}, k={k} => all equal: {np.all(flipped_arr == rot)}')

            if not np.all(flipped_arr == rot):
                print(flipped_arr)
                print(rot)
                print()
