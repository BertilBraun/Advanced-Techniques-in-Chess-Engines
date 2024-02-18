#pragma once

#include "common.hpp"

torch::Tensor encodeBoard(const Board &board) {
    // Encodes a chess board into a 12x8x8 numpy array.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // The first 6 layers represent the white pieces, and the last 6 layers
    // represent the black pieces.
    //
    // :param board: The chess board to encode.
    // :return: A 12x8x8 numpy array representing the encoded board.

    auto encodedBoard = torch::zeros({ENCODING_CHANNELS, 8, 8});

    for (Color color : COLORS) {
        for (PieceType pieceType : PIECE_TYPES) {
            int layerIndex = color * 6 + pieceType - 1;
            auto bitboard = board.piecesMask(pieceType, color);

            for (Square square : SQUARES) {
                int row = squareRank(square);
                int col = squareFile(square);
                if (bitboard & (1ULL << square)) {
                    encodedBoard[layerIndex][row][col] = 1;
                }
            }
        }
    }

    return encodedBoard;
}

torch::Tensor encodeBoards(const std::vector<Board> &boards) {
    // Encodes a list of chess boards into a Nx12x8x8 numpy array.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // The first 6 layers represent the current player's pieces, and the last 6 layers
    // represent the opponent's pieces. The layers are always oriented so that the current
    // player's pieces are at the bottom of the first dimension, and the opponent's pieces are
    // at the top.
    //
    // :param boards: The chess boards to encode.
    // :return: A Nx12x8x8 numpy array representing the encoded boards.

    std::vector<torch::Tensor> encodedBoards;
    for (const auto &board : boards) {
        encodedBoards.push_back(encodeBoard(board));
    }
    return torch::stack(encodedBoards);
}

torch::Tensor flipBoardHorizontal(const torch::Tensor &encodedBoard) {
    return encodedBoard.flip(2); // Flip along the width (columns)
}

torch::Tensor flipBoardVertical(const torch::Tensor &encodedBoard) {
    return encodedBoard.flip(1); // Flip along the height (rows)
}

float getBoardResultScore(Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is 1.0 if white has won, 0.0 if the game is a draw, and -1.0 if black has
    // won.
    //
    // :param board: The board to get the result score for.
    // :return: The result score for the given board.

    if (board.isCheckmate()) {
        return board.turn == WHITE ? -1.0f : 1.0f;
    } else if (board.isStalemate() || board.isInsufficientMaterial() ||
               board.isSeventyFiveMoves()) {
        return 0.0f;
    } else {
        throw std::runtime_error("Board is not in a terminal state");
    }
}
