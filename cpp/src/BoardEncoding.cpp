#include "BoardEncoding.hpp"

CompressedEncodedBoard encodeBoard(const Board *board) {
    // Encodes a chess board into a ENCODING_CHANNELSx8x8 tensor.
    //
    // Each layer in the first dimension represents one of the 12 distinct
    // piece types (6 for each color). Each cell in the 8x8 board for each layer
    // is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.
    //
    // On layer 12-15 are the castling right and layer 16 contains the en-passant square.
    //
    // param board: The chess board to encode.
    // :return: A ENCODING_CHANNELSx8x8 tensor representing the encoded board.

    TimeItGuard timer("encodeBoard");

    CompressedEncodedBoard encodedBoard{};

    const Board tmpBoard((board->currentPlayer() == -1) ? board->position().flip() : board->fen());
    const Position &tmp = tmpBoard.position();

    for (auto [i, color] : enumerate(COLORS)) {
        for (auto [j, piece_type] : enumerate(PIECE_TYPES)) {
            const Bitboard pieces = tmp.pieces(color, piece_type);
            const uint64 layerIndex = i * 6 + j;

            encodedBoard[layerIndex] = pieces;
        }
    }

    // The castling rights encoded
    constexpr uint64 allSet = 0xFFFFFFFFFFFFFFFF; // All bits set to 1
    encodedBoard[12] =
        allSet * tmp.can_castle(CastlingRights::WHITE_OO); // Kingside castling rights for white
    encodedBoard[13] =
        allSet * tmp.can_castle(CastlingRights::WHITE_OOO); // Queenside castling rights for white
    encodedBoard[14] =
        allSet * tmp.can_castle(CastlingRights::BLACK_OO); // Kingside castling rights for black
    encodedBoard[15] =
        allSet * tmp.can_castle(CastlingRights::BLACK_OOO); // Queenside castling rights for black

    if (tmp.ep_square()) {
        encodedBoard[16] = 1ULL << tmp.ep_square();
    }

    return encodedBoard;
}

torch::Tensor toTensor(const CompressedEncodedBoard &compressed, torch::Device device) {
    // Converts a compressed 64-bit array to an uncompressed ENCODING_CHANNELS x 8 x 8 tensor.
    //
    // param compressed: The 64-bit array to convert.
    // :return: The uncompressed tensor.

    TimeItGuard timer("toTensor");

    // Create tensor on CPU first
    torch::Tensor tensor = torch::zeros({ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH},
                                        torch::TensorOptions().dtype(torch::kUInt8));

    // Get CPU data pointer
    uint8_t* data = tensor.data_ptr<uint8_t>();

    // Fast CPU bit unpacking
    for (int channel = 0; channel < ENCODING_CHANNELS; ++channel) {
        const uint64_t bits = compressed[channel];
        uint8_t* channel_data = data + channel * BOARD_LENGTH * BOARD_LENGTH;

        for (int i = 0; i < 64; ++i) {
            channel_data[i] = (bits >> i) & 1;
        }
    }

    // Single GPU transfer
    return tensor.to(device);
}

std::optional<float> getBoardResultScore(const Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is -1.0 if the current player has lost, otherwise it must have been a
    // draw, in which case the score is based on the remaining material.
    //
    // param board: The board to get the result score for.
    // :return: The result score for the given board.

    if (!board.isGameOver())
        return std::nullopt;

    if (const auto winner = board.checkWinner()) {
        assert(winner.value() != board.currentPlayer());
        return {-1.0f}; // Our last move put us in checkmate
    }

    // Draw -> Return a score between based on the remaining material
    // The materialScore is positive if the white player has the advantage, and negative if the
    // black player has the advantage. Therefore, we need to negate the material score if the
    // black player is the current player.
    return 0.5 * board.currentPlayer() * board.approximateResultScore();
}