#include "BoardEncoding.hpp"

// Flips position with the white and black sides reversed.
std::string flip(const std::string &fen) {

    std::string f, token;
    std::stringstream ss(fen);

    for (Rank r = RANK_8; r >= RANK_1; --r) // Piece placement
    {
        std::getline(ss, token, r > RANK_1 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));
    }

    ss >> token;                       // Active color
    f += (token == "w" ? "B " : "W "); // Will be lowercased later

    ss >> token; // Castling availability
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token; // En passant square
    f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

    std::getline(ss, token); // Half and full moves
    f += token;

    return f;
}

CompressedEncodedBoard encodeBoard(const Board *board) {
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

    const Board tmpBoard((board->current_player() == -1) ? flip(board->fen()) : board->fen());
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

    torch::Tensor tensor = torch::zeros({ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH},
                                        torch::TensorOptions().dtype(torch::kUInt8).device(device));

    for (int channel = 0; channel < ENCODING_CHANNELS; ++channel) {
        const uint64 bits = compressed[channel];
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
    // param compressed: The 64-bit array to hash.
    // :return: The hash of the array.

    uint64 hash = 0;
    for (const int channel : range(ENCODING_CHANNELS)) {
        hash ^= compressed[channel] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

std::optional<float> getBoardResultScore(const Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is -1.0 if the current player has lost, otherwise it must have been a
    // draw, in which case the score is based on the remaining material.
    //
    // param board: The board to get the result score for.
    // :return: The result score for the given board.

    if (!board.is_game_over())
        return std::nullopt;

    if (const auto winner = board.check_winner()) {
        assert(winner.value() != board.current_player());
        return {-1.0f}; // Our last move put us in checkmate
    }

    // Draw -> Return a score between based on the remaining material
    // The materialScore is positive if the white player has the advantage, and negative if the
    // black player has the advantage. Therefore, we need to negate the material score if the
    // black player is the current player.
    return 0.5 * board.current_player() * board.get_approximate_result_score();
}