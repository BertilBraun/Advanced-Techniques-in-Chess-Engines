from Framework import *


ENCODING_CHANNELS = 6 + 6


def encode_board(board: Board) -> NDArray[np.float32]:
    """
    Encodes a chess board into a 12x8x8 numpy array.

    Each layer in the first dimension represents one of the 12 distinct
    piece types (6 for each color). Each cell in the 8x8 board for each layer
    is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.

    The first 6 layers represent the white pieces, and the last 6 layers
    represent the black pieces.

    :param board: The chess board to encode.
    :return: A 12x8x8 numpy array representing the encoded board.
    """
    encoded_board = np.zeros((12, 8, 8), dtype=np.float32)

    for color in COLORS:
        for piece_type in PIECE_TYPES:
            # Determine the index for this piece type and color
            color_offset = 0 if color == WHITE else 6
            layer_index = color_offset + piece_type - 1  # piece_type enum starts at 1 for PAWN

            # Get the bitboard for this piece type and color
            bitboard = board.pieces_mask(piece_type, color)

            # Convert bitboard to board positions
            for square in SQUARES:
                row, col = divmod(square, 8)
                encoded_board[layer_index, col, row] = (bitboard & BB_SQUARES[square]) != 0

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


def flip_board_horizontal(encoded_board: NDArray[np.float32]) -> NDArray[np.float32]:
    # Flip along the 2nd axis (columns of the 8x8 grid for each piece type)
    return np.flip(encoded_board, axis=2)


def flip_board_vertical(encoded_board: NDArray[np.float32]) -> NDArray[np.float32]:
    # Flip along the 1st axis (rows of the 8x8 grid for each piece type)
    return np.flip(encoded_board, axis=1)


def get_board_result_score(board: Board) -> float:
    """
    Returns the result score for the given board.

    The result score is 1.0 if white has won, 0.0 if the game is a draw, and -1.0 if black has won.

    :param board: The board to get the result score for.
    :return: The result score for the given board.
    """
    if board.is_checkmate():
        return -1.0 if board.turn == WHITE else 1.0
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0.0
    else:
        raise ValueError('Board is not in a terminal state')
