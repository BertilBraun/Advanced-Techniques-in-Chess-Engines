#pragma once

#include "common.hpp"

#include <cstdint>

// 6 types for each color + 1 for castling and en passant + 1 for current player
static inline constexpr int ENCODING_CHANNELS = 6 + 6 + 2;

typedef std::array<std::array<std::array<int8, BOARD_LENGTH>, BOARD_LENGTH>, ENCODING_CHANNELS>
    EncodedBoard;
typedef std::array<uint64, ENCODING_CHANNELS> CompressedEncodedBoard;

CompressedEncodedBoard encodeBoard(const Board &board) {
    // Encodes a chess board into a 14x8x8 tensor.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // On layer 12 are the castling right and 13 a color indicator.
    //
    // :param board: The chess board to encode.
    // :return: A 14x8x8 tensor representing the encoded board.

    CompressedEncodedBoard encodedBoard{};

    for (auto [i, color] : enumerate(COLORS)) {
        for (auto [j, piece_type] : enumerate(PIECE_TYPES)) {
            const Bitboard pieces = board.piecesMask(piece_type, color);
            const uint64 layerIndex = i * 6 + j;

            encodedBoard[layerIndex] = pieces;
        }
    }

    encodedBoard[12] = ( // The castling rights encoded in the corners of a bitboard
        board.hasKingsideCastlingRights(WHITE) << (0) |                 // Upper left corner
        board.hasQueensideCastlingRights(WHITE) << (BOARD_LENGTH - 1) | // Upper right corner
        board.hasKingsideCastlingRights(BLACK)
            << (BOARD_SIZE - BOARD_LENGTH) |                        // Bottom left corner
        board.hasQueensideCastlingRights(BLACK) << (BOARD_SIZE - 1) // Bottom right corner
    );

    if (board.ep_square.has_value()) {
        Square square = board.ep_square.value();
        assert(square != 0 && square != 7 && square != 56 && square != 63);
        encodedBoard[12] |= (1 << square);
    }

    encodedBoard[13] = (board.turn == WHITE) ? 0xFFFFFFFFFFFFFFFF : 0;

    // TODO: NOTE: previously this was then first decoded into binary and flipped along axis=1, but
    // I think both is unnecessary

    return encodedBoard;
}

CompressedEncodedBoard compress(const EncodedBoard &binary) {
    // Converts a binary array to a compressed 64-bit array.
    //
    // :param binary: The binary array to convert.
    // :return: The 64-bit array.

    CompressedEncodedBoard compressed{};

    for (int channel : range(ENCODING_CHANNELS)) {
        for (int row : range(BOARD_LENGTH)) {
            for (int col : range(BOARD_LENGTH)) {
                compressed[channel] |= (uint64) binary[channel][row][col]
                                       << (row * BOARD_LENGTH + col);
            }
        }
    }

    return compressed;
}

EncodedBoard decompress(const CompressedEncodedBoard &compressed) {
    // Converts a compressed 64-bit array to a binary array.
    //
    // :param compressed: The 64-bit array to convert.
    // :return: The binary array.

    EncodedBoard binary{};

    for (int channel : range(ENCODING_CHANNELS)) {
        for (int row : range(BOARD_LENGTH)) {
            for (int col : range(BOARD_LENGTH)) {
                binary[channel][row][col] = (compressed[channel] >> (row * BOARD_LENGTH + col)) & 1;
            }
        }
    }

    return binary;
}

uint64 hash(const CompressedEncodedBoard &compressed) {
    // Computes the hash of a compressed 64-bit array.
    //
    // :param compressed: The 64-bit array to hash.
    // :return: The hash of the array.

    uint64 hash = 0;
    for (int channel : range(ENCODING_CHANNELS)) {
        hash ^= compressed[channel] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

Board decodeBoard(const EncodedBoard &encodedBoard) {
    // Decodes a 12x8x8 tensor into a chess board.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // The board is always oriented so that the current player's pieces are at the bottom of the
    // first dimension, and the opponent's pieces are at the top. The first 6 layers represent
    // the current player's pieces, and the last 6 layers represent the opponent's pieces. The
    // current player is always white.
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
                if (layer[row][col]) {
                    board.setPieceAt(square, Piece(pieceType, color));
                }
            }
        }
    }

    return board;
}

static inline std::array<float, PieceType::NUM_PIECE_TYPES> __PIECE_VALUES = {
    0.0f, // None
    1.0f, // Pawn
    3.0f, // Knight
    3.0f, // Bishop
    5.0f, // Rook
    9.0f, // Queen
    0.0f  // King
};

inline float __MAX_MATERIAL_SCORE =
    __PIECE_VALUES[PieceType::QUEEN] + 2 * __PIECE_VALUES[PieceType::ROOK] +
    2 * __PIECE_VALUES[PieceType::BISHOP] + 2 * __PIECE_VALUES[PieceType::KNIGHT] +
    8 * __PIECE_VALUES[PieceType::PAWN];

float getMaterialScore(const Board &board) {
    // Returns the material score for the given board. The score is positive if the white player
    // has the advantage, and negative if the black player has the advantage. The score is
    // normalized to be in the range [-1.0, 1.0].

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
    // The result score is 1.0 if the current player has won, otherwise it must have been a
    // draw, in which case the score is based on the remaining material.
    //
    // :param board: The board to get the result score for.
    // :return: The result score for the given board.

    if (!board.isGameOver())
        return std::nullopt;

    if (board.isCheckmate())
        return {-1.0f}; // Our last move put us in checkmate

    // Draw -> Return a score between based on the remaining material
    // The materialScore is positive if the white player has the advantage, and negative if the
    // black player has the advantage. Therefore, we need to negate the material score if the
    // black player is the current player.
    float materialScore = getMaterialScore(board);
    if (board.turn == BLACK)
        materialScore = -materialScore;

    return {materialScore};
}
