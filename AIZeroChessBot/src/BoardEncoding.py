from Framework import *


ENCODING_CHANNELS = 6 + 6


def encode_board(board: Board) -> NDArray[np.float32]:
    """
    Encodes a chess board into a 12x8x8 numpy array.

    Each layer in the first dimension represents one of the 12 distinct
    piece types (6 for each color). Each cell in the 8x8 board for each layer
    is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.

    The first 6 layers represent the current player's pieces, and the last 6 layers
    represent the opponent's pieces. The layers are always oriented so that the current
    player's pieces are at the bottom of the first dimension, and the opponent's pieces are
    at the top.

    :param board: The chess board to encode.
    :return: A 12x8x8 numpy array representing the encoded board.
    """
    encoded_board = np.zeros((12, 8, 8), dtype=np.float32)

    for color in COLORS:
        for piece_type in PIECE_TYPES:
            # Determine the index for this piece type and color
            color_offset = 0 if color == board.turn else 6
            layer_index = color_offset + piece_type - 1  # piece_type enum starts at 1 for PAWN

            # Get the bitboard for this piece type and color
            bitboard = board.pieces_mask(piece_type, color)

            if board.turn == BLACK:
                bitboard = flip_vertical(bitboard)

            # Convert bitboard to board positions
            for square in SQUARES:
                row, col = divmod(square, 8)
                encoded_board[layer_index, row, col] = (bitboard & BB_SQUARES[square]) != 0

    return encoded_board


def encode_boards(boards: list[Board]) -> NDArray[np.float32]:
    """
    Encodes a list of chess boards into a Nx12x8x8 numpy array.

    Each layer in the first dimension represents one of the 12 distinct
    piece types (6 for each color). Each cell in the 8x8 board for each layer
    is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.

    The first 6 layers represent the current player's pieces, and the last 6 layers
    represent the opponent's pieces. The layers are always oriented so that the current
    player's pieces are at the bottom of the first dimension, and the opponent's pieces are
    at the top.

    :param boards: The chess boards to encode.
    :return: A Nx12x8x8 numpy array representing the encoded boards.
    """
    return np.stack([encode_board(board) for board in boards])


def get_board_result_score(board: Board, current_player: Color) -> float:
    """
    Returns the result score for the given board from the perspective of the current player after he has made a move.

    The result score is 1.0 if the current player has won, 0.0 if the game is a draw, and -1.0 if the current player has lost.

    :param board: The board to get the result score for.
    :param current_player: The current player's color.
    :return: The result score for the given board from the perspective of the current player.
    """
    if board.is_checkmate():
        return -1.0 if board.turn == current_player else 1.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0.0
    else:
        raise ValueError('Board is not in a terminal state')


if __name__ == '__main__':
    board = Board()
    enc1 = encode_board(board)
    print(enc1)
    print('-' * 100)
    board.push(Move.null())
    enc2 = encode_board(board)
    print(enc2)
    print('-' * 100)
    print(enc1 == enc2)
