#pragma once

#include "common.hpp"

torch::Tensor encodeBoard(const Board &board, torch::Device device) {
    // Encodes a chess board into a 12x8x8 tensor.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // The first 6 layers represent the current players pieces, and the last 6 layers
    // represent the opponents pieces. The pieces are always oriented so that the current
    // player's pieces are at the bottom of the first dimension, and the opponent's pieces are
    // at the top.
    //
    // :param board: The chess board to encode.
    // :return: A 12x8x8 tensor representing the encoded board.

    auto encodedBoard = torch::zeros({ENCODING_CHANNELS, 8, 8},
                                     torch::TensorOptions().device(device).dtype(torch::kFloat16));

    for (Color color : COLORS) {
        for (PieceType pieceType : PIECE_TYPES) {
            int layerIndex = (color != board.turn) * 6 + pieceType - 1;
            auto bitboard = board.piecesMask(pieceType, color);

            for (Square square : SQUARES) {
                if (board.turn == BLACK)
                    square = squareFlipVertical(square);

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

torch::Tensor encodeBoards(const std::vector<Board> &boards, torch::Device device) {
    // Encodes a list of chess boards into a Nx12x8x8 tensor.
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
    // :return: A Nx12x8x8 tensor representing the encoded boards.

    std::vector<torch::Tensor> encodedBoards;
    for (const auto &board : boards) {
        encodedBoards.push_back(encodeBoard(board, device));
    }
    return torch::stack(encodedBoards);
}

Board decodeBoard(const torch::Tensor &encodedBoard) {
    // Decodes a 12x8x8 tensor into a chess board.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // The board is always oriented so that the current player's pieces are at the bottom of the
    // first dimension, and the opponent's pieces are at the top. The first 6 layers represent the
    // current player's pieces, and the last 6 layers represent the opponent's pieces. The current
    // player is always white.
    //
    // :param encodedBoard: The 12x8x8 tensor to decode.
    // :return: The decoded chess board.

    Board board(false);

    for (Color color : COLORS) {
        for (PieceType pieceType : PIECE_TYPES) {
            int layerIndex = color * 6 + pieceType - 1;
            auto layer = encodedBoard[layerIndex];

            for (Square square : SQUARES) {
                int row = squareRank(square);
                int col = squareFile(square);
                if (layer[row][col].item<float>() > 0.5) {
                    board.setPieceAt(square, Piece(pieceType, color));
                }
            }
        }
    }

    return board;
}

std::array<float, PieceType::NUM_PIECE_TYPES> __PIECE_VALUES = {
    0.0f, // None
    1.0f, // Pawn
    3.0f, // Knight
    3.0f, // Bishop
    5.0f, // Rook
    9.0f, // Queen
    0.0f  // King
};

float __MAX_MATERIAL_SCORE =
    __PIECE_VALUES[PieceType::QUEEN] + 2 * __PIECE_VALUES[PieceType::ROOK] +
    2 * __PIECE_VALUES[PieceType::BISHOP] + 2 * __PIECE_VALUES[PieceType::KNIGHT] +
    8 * __PIECE_VALUES[PieceType::PAWN];

float getMaterialScore(const Board &board) {
    // Returns the material score for the given board. The score is positive if the white player has
    // the advantage, and negative if the black player has the advantage. The score is normalized to
    // be in the range [-1.0, 1.0].

    float materialScore = 0.0f;
    for (PieceType pieceType : PIECE_TYPES) {
        int whitePieceCount = board.pieces(pieceType, WHITE).size();
        int blackPieceCount = board.pieces(pieceType, BLACK).size();
        int pieceValue = __PIECE_VALUES[pieceType];
        materialScore += (whitePieceCount - blackPieceCount) * pieceValue;
    }

    return materialScore / __MAX_MATERIAL_SCORE;
}

std::optional<float> getBoardResultScore(Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is 1.0 if the current player has won, otherwise it must have been a draw,
    // in which case the score is between -0.5 and 0.5 based on the remaining material.
    //
    // :param board: The board to get the result score for.
    // :return: The result score for the given board.

    if (!board.isGameOver())
        return std::nullopt;

    if (board.isCheckmate())
        return {-1.0f}; // Our last move put us in checkmate

    // Draw -> Return a score between -0.5 and 0.5 based on the remaining material
    // The materialScore is positive if the white player has the advantage, and negative if the
    // black player has the advantage. Therefore, we need to negate the material score if the black
    // player is the current player.
    float materialScore = getMaterialScore(board);
    if (board.turn == BLACK)
        materialScore = -materialScore;

    // Normalize the material score to be in the range [-0.5, 0.5]
    return {0.5f * materialScore};
}
