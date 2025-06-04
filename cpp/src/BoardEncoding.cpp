#include "BoardEncoding.hpp"

CompressedEncodedBoard encodeBoard(const Board &board) {
    // Encodes a chess board into a ENCODING_CHANNELSx8x8 tensor.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // On layer 12-15 are the castling right and layer 16 contains the en-passant square.
    //
    // :param board: The chess board to encode.
    // :return: A ENCODING_CHANNELSx8x8 tensor representing the encoded board.

    CompressedEncodedBoard encodedBoard{};

    for (auto [i, color] : enumerate(COLORS)) {
        for (auto [j, piece_type] : enumerate(PIECE_TYPES)) {
            const Bitboard pieces = board.piecesMask(piece_type, color);
            const uint64 layerIndex = i * 6 + j;

            encodedBoard[layerIndex] = pieces;
        }
    }

    // The castling rights encoded in the corners of a bitboard
    encodedBoard[12] = (
        // Upper left corner
        (uint64) board.hasKingsideCastlingRights(WHITE) << (0) |
        // Upper right corner
        (uint64) board.hasQueensideCastlingRights(WHITE) << (BOARD_LENGTH - 1) |
        // Bottom left corner
        (uint64) board.hasKingsideCastlingRights(BLACK) << (BOARD_SIZE - BOARD_LENGTH) |
        // Bottom right corner
        (uint64) board.hasQueensideCastlingRights(BLACK) << (BOARD_SIZE - 1));

    if (board.ep_square.has_value()) {
        Square square = board.ep_square.value();
        assert(square != 0 && square != 7 && square != 56 && square != 63);
        encodedBoard[12] |= (((uint64) 1) << square);
    }

    encodedBoard[13] = (board.turn == WHITE) ? 0xFFFFFFFFFFFFFFFF : 0;

    return encodedBoard;
}

torch::Tensor toTensor(const CompressedEncodedBoard &compressed, torch::Device device) {
    // Converts a compressed 64-bit array to a uncompressed ENCODING_CHANNELS x 8 x 8 tensor.
    //
    // :param compressed: The 64-bit array to convert.
    // :return: The uncompressed tensor.

    torch::Tensor tensor =
        torch::zeros({ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH}, torch::kUInt8);

    for (int channel = 0; channel < ENCODING_CHANNELS; ++channel) {
        uint64 bits = compressed[channel];
        for (int row = 0; row < BOARD_LENGTH; ++row) {
            for (int col = 0; col < BOARD_LENGTH; ++col) {
                if ((bits >> (row * BOARD_LENGTH + col)) & 1) {
                    tensor[channel][row][col] = 1;
                }
            }
        }
    }

    return tensor;
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

static const std::unordered_map<Stockfish::PieceType, float> PIECE_VALUES = {
    0.0f, // None
    1.0f, // Pawn
    3.0f, // Knight
    3.0f, // Bishop
    5.0f, // Rook
    9.0f, // Queen
    0.0f  // King
};

const float MAX_MATERIAL_SCORE =
    PIECE_VALUES[PieceType::QUEEN] + 2 * PIECE_VALUES[PieceType::ROOK] +
    2 * PIECE_VALUES[PieceType::BISHOP] + 2 * PIECE_VALUES[PieceType::KNIGHT] +
    8 * PIECE_VALUES[PieceType::PAWN];

float getMaterialScore(const Board &board) {
    // Returns the material score for the given board. The score is positive if the white player
    // has the advantage, and negative if the black player has the advantage. The score is
    // normalized to be in the range [-1.0, 1.0].

    float materialScore = 0.0f;
    for (PieceType pieceType : PIECE_TYPES) {
        int whitePieceCount = board.pieces(pieceType, WHITE).size();
        int blackPieceCount = board.pieces(pieceType, BLACK).size();
        int pieceValue = PIECE_VALUES[pieceType];
        materialScore += (whitePieceCount - blackPieceCount) * pieceValue;
    }

    return materialScore / MAX_MATERIAL_SCORE;
}
std::optional<float> getBoardResultScore(Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is -1.0 if the current player has lost, otherwise it must have been a
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