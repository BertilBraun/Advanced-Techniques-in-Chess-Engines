#include "BoardEncoding.hpp"

constexpr int pieceCount(const Bitboard bb) noexcept { return std::popcount(bb); }

constexpr uint64 splitmix64(uint64 x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::size_t BoardHash::operator()(CompressedEncodedBoard const &b) const noexcept {
    uint64 h = 0xcbf29ce484222325ull;
    for (const uint64 v : b.bits)
        h = splitmix64(h ^ splitmix64(v));
    for (const int8 v : b.scal)
        h = splitmix64(h ^ static_cast<uint64>(v));
    return static_cast<size_t>(h);
}

CompressedEncodedBoard encodeBoard(const Board *board) {
    TIMEIT("encodeBoard");
    CompressedEncodedBoard out{};

    // ---- 1) normalise position so it's *always white to move* -------------
    const Board tmpBoard((board->currentPlayer() == -1) ? board->position().flip() : board->fen());
    const Position &tmp = tmpBoard.position();

    // ---- 2) piece-type channels -------------------------------------------
    int ch = 0;
    for (const Color color : {WHITE, BLACK}) {
        for (const PieceType piece : PIECE_TYPES) {
            out.bits[ch++] = tmp.pieces(color, piece);
        }
    }

    // ---- 3) castling rights ------------------------------------------------
    constexpr uint64 ALL_SET = 0xFFFF'FFFF'FFFF'FFFFull;
    out.bits[ch++] = ALL_SET * tmp.can_castle(CastlingRights::WHITE_OO);
    out.bits[ch++] = ALL_SET * tmp.can_castle(CastlingRights::WHITE_OOO);
    out.bits[ch++] = ALL_SET * tmp.can_castle(CastlingRights::BLACK_OO);
    out.bits[ch++] = ALL_SET * tmp.can_castle(CastlingRights::BLACK_OOO);

    // ---- 4) occupancy planes ----------------------------------------------
    out.bits[ch++] = tmp.pieces(Color::WHITE);
    out.bits[ch++] = tmp.pieces(Color::BLACK);

    // ---- 5) “checkers” mask (attackers of side-to-move’s king) ------------
    out.bits[ch++] = tmp.checkers();

    static_assert(BINARY_C == 19);

    // ---- 6) material-difference scalars -----------------------------------
    for (int i = 0; i < 6; ++i) {
        const Bitboard white = tmp.pieces(WHITE, PIECE_TYPES[i]);
        const Bitboard black = tmp.pieces(BLACK, PIECE_TYPES[i]);
        out.scal[i] = static_cast<int8>(pieceCount(white) - pieceCount(black));
    }

    return out;
}

torch::Tensor toTensor(const CompressedEncodedBoard &compressed) {
    TIMEIT("toTensor");

    auto t =
        torch::empty({BOARD_C, BOARD_LEN, BOARD_LEN}, torch::TensorOptions().dtype(torch::kInt8));

    int8 *dst = t.data_ptr<int8>();

    // -------- binary planes -------------------------------------------------
    for (int ch = 0; ch < BINARY_C; ++ch) {
        const uint64 bits = compressed.bits[ch];
        int8 *d = dst + ch * 64;

        // unroll for speed: eight bytes at a time
        for (int byte = 0; byte < 8; ++byte) {
            const uint8 b = (bits >> (byte * 8)) & 0xFFu;
            // expand 8 bits → 8 bytes
            d[byte * 8 + 0] = b & 1;
            d[byte * 8 + 1] = (b >> 1) & 1;
            d[byte * 8 + 2] = (b >> 2) & 1;
            d[byte * 8 + 3] = (b >> 3) & 1;
            d[byte * 8 + 4] = (b >> 4) & 1;
            d[byte * 8 + 5] = (b >> 5) & 1;
            d[byte * 8 + 6] = (b >> 6) & 1;
            d[byte * 8 + 7] = (b >> 7) & 1;
        }
    }

    // -------- scalar planes -------------------------------------------------
    for (int i = 0; i < SCALAR_C; ++i) {
        int8 *d = dst + (BINARY_C + i) * 64;
        std::memset(d, compressed.scal[i], 64); // broadcast 1 byte → 64 bytes
    }

    return t;
}

float getBoardResultScore(const Board &board) {
    // Returns the result score for the given board.
    //
    // The result score is -1.0 if the current player has lost, otherwise it must have been a
    // draw, in which case the score is based on the remaining material.
    //
    // param board: The board to get the result score for.
    // :return: The result score for the given board.

    assert(board.isGameOver() && "Game was not over!");

    if (const auto winner = board.checkWinner()) {
        assert(winner.value() != board.currentPlayer());
        return -1.0f; // Our last move put us in checkmate
    }

    // Draw -> Return a score between based on the remaining material
    // The materialScore is positive if the white player has the advantage, and negative if the
    // black player has the advantage. Therefore, we need to negate the material score if the
    // black player is the current player.
    return 0.5 * board.currentPlayer() * board.approximateResultScore();
}