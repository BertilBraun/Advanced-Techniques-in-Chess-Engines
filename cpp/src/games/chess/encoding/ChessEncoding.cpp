#include "games/chess/encoding/ChessEncoding.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>

using namespace Stockfish;

namespace {
constexpr std::array pieceTypes = {
    Stockfish::PieceType::PAWN, Stockfish::PieceType::KNIGHT, Stockfish::PieceType::BISHOP,
    Stockfish::PieceType::ROOK, Stockfish::PieceType::QUEEN,  Stockfish::PieceType::KING,
};
constexpr std::uint64_t allSquares = 0xFFFF'FFFF'FFFF'FFFFULL;
constexpr std::uint64_t checkerboard = 0xAA55'AA55'AA55'AA55ULL;
}

constexpr int pieceCount(const Bitboard bb) noexcept { return std::popcount(bb); }

constexpr Bitboard flipRanks(const Bitboard bits) noexcept {
    return ((bits & 0x0000'0000'0000'00FFULL) << 56) | ((bits & 0x0000'0000'0000'FF00ULL) << 40) |
           ((bits & 0x0000'0000'00FF'0000ULL) << 24) | ((bits & 0x0000'0000'FF00'0000ULL) << 8) |
           ((bits & 0x0000'00FF'0000'0000ULL) >> 8) | ((bits & 0x0000'FF00'0000'0000ULL) >> 24) |
           ((bits & 0x00FF'0000'0000'0000ULL) >> 40) | ((bits & 0xFF00'0000'0000'0000ULL) >> 56);
}

CompressedEncodedBoard encodeBoard(const Board &board) {
    CompressedEncodedBoard out{};

    // Lc0's INPUT_CLASSICAL_112_PLANE layout. Every plane is written from the side to move's
    // perspective: ranks are mirrored for Black and the two colours swap, which is the same
    // canonical convention this project already used.
    const Position &position = board.position();
    const bool flipForBlack = position.side_to_move() == BLACK;
    const auto canonicalBits = [flipForBlack](const Bitboard bits) {
        return BitBoard<ChessRepresentationDimensions::boardLength>(
            {flipForBlack ? flipRanks(bits) : bits});
    };

    int ch = 0;
    const auto writePlacement = [&](const std::array<std::uint64_t, 12> &pieces,
                                    const bool repeated) {
        // Indices 0-5 are White's pieces and 6-11 are Black's; "ours" comes first for the
        // side to move, so Black's planes lead when Black is to move.
        const int ownBase = flipForBlack ? 6 : 0;
        const int opponentBase = flipForBlack ? 0 : 6;
        for (const int offset : range(6)) {
            out.binaryPlanes[ch++] = canonicalBits(pieces[ownBase + offset]);
        }
        for (const int offset : range(6)) {
            out.binaryPlanes[ch++] = canonicalBits(pieces[opponentBase + offset]);
        }
        out.binaryPlanes[ch++] = canonicalBits(repeated ? allSquares : 0ULL);
    };

    writePlacement(Board::placementOf(position).pieces, board.repetitionCount() >= 1);
    for (const Board::PiecePlacement &placement : board.previousPlacements()) {
        if (placement.occupied) {
            writePlacement(placement.pieces, placement.repeated);
        } else {
            // Before the start of the recorded game Lc0 leaves the history planes at zero.
            ch += ChessRepresentationDimensions::planesPerHistoryPosition;
        }
    }

    const Color ownColor = position.side_to_move();
    const Color opponentColor = ~ownColor;
    const auto castlingPlane = [&](const Color color, const CastlingSide side) {
        return canonicalBits(allSquares * (position.can_castle(color & side) != 0));
    };
    out.binaryPlanes[ch++] = castlingPlane(ownColor, QUEEN_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(ownColor, KING_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(opponentColor, QUEEN_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(opponentColor, KING_SIDE);
    out.binaryPlanes[ch++] =
        BitBoard<ChessRepresentationDimensions::boardLength>({flipForBlack ? allSquares : 0ULL});

    assert(ch == ChessRepresentationDimensions::binaryChannelCount);

    // The wrapper module divides this by 99; it is kept as a raw count because the input tensor
    // reaching the network is int8.
    out.scalarPlanes[0] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));
    out.scalarPlanes[1] = 0;
    out.scalarPlanes[2] = 1;

    return out;
}

torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed) {
    auto tensor = torch::empty({ChessRepresentationDimensions::channelCount,
                                ChessRepresentationDimensions::boardLength,
                                ChessRepresentationDimensions::boardLength},
                               torch::TensorOptions().dtype(torch::kInt8));

    compressed.writeTensorInto(std::span<std::int8_t>(tensor.data_ptr<std::int8_t>(),
                                                      CompressedEncodedBoard::tensorValues));
    return tensor;
}

void ChessEncoding::encodeInputInto(const Board &state, std::int8_t *destination) {
    const CompressedEncodedBoard encoded = encodeBoard(state);
    encoded.writeTensorInto(
        std::span<std::int8_t>(destination, CompressedEncodedBoard::tensorValues));
}
