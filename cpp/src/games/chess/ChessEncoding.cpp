#include "games/chess/ChessEncoding.hpp"

#include "games/chess/ChessState.hpp"

#include <algorithm>
#include <array>
#include <bitboard.h>
#include <cstring>
#include <position.h>

namespace az::games::chess {
namespace {

constexpr uint64 ALL_SQUARES = 0xFFFFFFFFFFFFFFFFULL;
constexpr std::array<Stockfish::PieceType, 6> PIECE_TYPES{
    Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
    Stockfish::ROOK, Stockfish::QUEEN,  Stockfish::KING,
};

[[nodiscard]] constexpr uint64 flipRanks(uint64 bits) {
    return ((bits & 0x00000000000000FFULL) << 56U) | ((bits & 0x000000000000FF00ULL) << 40U) |
           ((bits & 0x0000000000FF0000ULL) << 24U) | ((bits & 0x00000000FF000000ULL) << 8U) |
           ((bits & 0x000000FF00000000ULL) >> 8U) | ((bits & 0x0000FF0000000000ULL) >> 24U) |
           ((bits & 0x00FF000000000000ULL) >> 40U) | ((bits & 0xFF00000000000000ULL) >> 56U);
}

void writeBinaryPlane(ChessEncoding &encoding, int32 plane, uint64 bits) {
    const std::size_t offset = static_cast<std::size_t>(plane) * 64U;
    for (int32 square = 0; square < 64; ++square) {
        encoding.values[offset + static_cast<std::size_t>(square)] =
            static_cast<int8>((bits >> square) & 1ULL);
    }
}

void writeScalarPlane(ChessEncoding &encoding, int32 plane, int8 value) {
    const auto first = encoding.values.begin() + static_cast<std::ptrdiff_t>(plane * 64);
    std::fill(first, first + 64, value);
}

} // namespace

int8 ChessEncoding::at(int32 plane, int32 row, int32 column) const {
    if (plane < 0 || plane >= CHESS_ENCODING_PLANES || row < 0 || row >= CHESS_BOARD_SIZE ||
        column < 0 || column >= CHESS_BOARD_SIZE) {
        throw std::out_of_range("chess encoding coordinate is outside the tensor");
    }
    return values[static_cast<std::size_t>(plane * 64 + row * 8 + column)];
}

ChessEncoding canonicalEncoding(const ChessState &state) {
    ChessEncoding encoding{};
    const Stockfish::Position &position = state.position();
    const bool flipForBlack = position.side_to_move() == Stockfish::BLACK;
    const auto canonicalBits = [flipForBlack](uint64 bits) {
        return flipForBlack ? flipRanks(bits) : bits;
    };

    int32 plane = 0;
    for (const Stockfish::Color outputColor : {Stockfish::WHITE, Stockfish::BLACK}) {
        const Stockfish::Color sourceColor = flipForBlack ? ~outputColor : outputColor;
        for (const Stockfish::PieceType pieceType : PIECE_TYPES) {
            writeBinaryPlane(encoding, plane++,
                             canonicalBits(position.pieces(sourceColor, pieceType)));
        }
    }
    for (const Stockfish::Color outputColor : {Stockfish::WHITE, Stockfish::BLACK}) {
        const Stockfish::Color sourceColor = flipForBlack ? ~outputColor : outputColor;
        writeBinaryPlane(encoding, plane++,
                         ALL_SQUARES * position.can_castle(sourceColor & Stockfish::KING_SIDE));
        writeBinaryPlane(encoding, plane++,
                         ALL_SQUARES * position.can_castle(sourceColor & Stockfish::QUEEN_SIDE));
    }
    for (const Stockfish::Color outputColor : {Stockfish::WHITE, Stockfish::BLACK}) {
        const Stockfish::Color sourceColor = flipForBlack ? ~outputColor : outputColor;
        writeBinaryPlane(encoding, plane++, canonicalBits(position.pieces(sourceColor)));
    }
    writeBinaryPlane(encoding, plane++, canonicalBits(position.checkers()));
    const Stockfish::Square enPassant = position.ep_square();
    writeBinaryPlane(
        encoding, plane++,
        enPassant == Stockfish::SQ_NONE ? 0ULL : canonicalBits(Stockfish::square_bb(enPassant)));
    writeBinaryPlane(encoding, plane++, ALL_SQUARES * (state.repetitionCount() >= 1));
    writeBinaryPlane(encoding, plane++, ALL_SQUARES * (state.repetitionCount() >= 2));
    assert(plane == 22);

    const Stockfish::Color whiteSource = flipForBlack ? Stockfish::BLACK : Stockfish::WHITE;
    const Stockfish::Color blackSource = flipForBlack ? Stockfish::WHITE : Stockfish::BLACK;
    for (const Stockfish::PieceType pieceType : PIECE_TYPES) {
        const int32 difference = Stockfish::popcount(position.pieces(whiteSource, pieceType)) -
                                 Stockfish::popcount(position.pieces(blackSource, pieceType));
        writeScalarPlane(encoding, plane++, static_cast<int8>(difference));
    }
    writeScalarPlane(
        encoding, plane++,
        static_cast<int8>(std::min(position.rule50_count(), state.rules().halfmoveDrawPlyCount)));
    assert(plane == CHESS_ENCODING_PLANES);
    return encoding;
}

ChessEncoding ChessState::canonicalEncoding() const { return chess::canonicalEncoding(*this); }

} // namespace az::games::chess
