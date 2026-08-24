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

    // Canonical chess inputs are always encoded from the side-to-move perspective.
    const Position &position = board.position();
    const bool flipForBlack = position.side_to_move() == BLACK;
    const auto canonicalBits = [flipForBlack](const Bitboard bits) {
        return BitBoard<ChessRepresentationDimensions::boardLength>(
            {flipForBlack ? flipRanks(bits) : bits});
    };

    int ch = 0;
    // Piece-type planes remain Stockfish bitboards until the encoding boundary.
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        for (const PieceType piece : pieceTypes) {
            out.binaryPlanes[ch++] = canonicalBits(position.pieces(positionColor, piece));
        }
    }

    constexpr std::uint64_t ALL_SET = 0xFFFF'FFFF'FFFF'FFFFull;
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binaryPlanes[ch++] =
            canonicalBits(ALL_SET * position.can_castle(positionColor & KING_SIDE));
        out.binaryPlanes[ch++] =
            canonicalBits(ALL_SET * position.can_castle(positionColor & QUEEN_SIDE));
    }

    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binaryPlanes[ch++] = canonicalBits(position.pieces(positionColor));
    }

    out.binaryPlanes[ch++] = canonicalBits(position.checkers());

    const Square epSquare = position.ep_square();
    out.binaryPlanes[ch++] = canonicalBits(epSquare == SQ_NONE ? 0ULL : square_bb(epSquare));
    const int repetitions = board.repetitionCount();
    out.binaryPlanes[ch++] = canonicalBits(ALL_SET * (repetitions >= 1));
    out.binaryPlanes[ch++] = canonicalBits(ALL_SET * (repetitions >= 2));

    assert(ch == ChessRepresentationDimensions::binaryChannelCount);

    for (const int i : range(6)) {
        const Color whiteSource = flipForBlack ? BLACK : WHITE;
        const Color blackSource = flipForBlack ? WHITE : BLACK;
        const Bitboard white = position.pieces(whiteSource, pieceTypes[i]);
        const Bitboard black = position.pieces(blackSource, pieceTypes[i]);
        out.scalarPlanes[i] = static_cast<std::int8_t>(pieceCount(white) - pieceCount(black));
    }
    out.scalarPlanes[6] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));

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
