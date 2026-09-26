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

ProjectEncodedBoard encodeProjectBoard(const Board &board) {
    ProjectEncodedBoard out{};

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

    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binaryPlanes[ch++] =
            canonicalBits(allSquares * position.can_castle(positionColor & KING_SIDE));
        out.binaryPlanes[ch++] =
            canonicalBits(allSquares * position.can_castle(positionColor & QUEEN_SIDE));
    }

    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binaryPlanes[ch++] = canonicalBits(position.pieces(positionColor));
    }

    out.binaryPlanes[ch++] = canonicalBits(position.checkers());

    const Square epSquare = position.ep_square();
    out.binaryPlanes[ch++] = canonicalBits(epSquare == SQ_NONE ? 0ULL : square_bb(epSquare));
    const int repetitions = board.repetitionCount();
    out.binaryPlanes[ch++] = canonicalBits(allSquares * (repetitions >= 1));
    out.binaryPlanes[ch++] = canonicalBits(allSquares * (repetitions >= 2));

    const auto &recentMoves = board.recentMoves();
    for (const std::size_t index : range(Board::RECENT_MOVE_COUNT)) {
        const bool present = index < board.recentMoveCount();
        out.binaryPlanes[ch++] = canonicalBits(present ? square_bb(recentMoves[index].from) : 0ULL);
        out.binaryPlanes[ch++] = canonicalBits(present ? square_bb(recentMoves[index].to) : 0ULL);
    }

    out.binaryPlanes[ch++] = BitBoard<ChessRepresentationDimensions::boardLength>({checkerboard});
    const Bitboard bishops = position.pieces(BISHOP);
    const bool oppositeColoredBishops = pieceCount(bishops) == 2 &&
                                        pieceCount(position.pieces(WHITE, BISHOP)) == 1 &&
                                        pieceCount(position.pieces(BLACK, BISHOP)) == 1 &&
                                        pieceCount(bishops & checkerboard) == 1;
    out.binaryPlanes[ch++] = canonicalBits(allSquares * oppositeColoredBishops);

    assert(ch == ProjectInputDimensions::binaryChannelCount);

    for (const int i : range(6)) {
        const Color whiteSource = flipForBlack ? BLACK : WHITE;
        const Color blackSource = flipForBlack ? WHITE : BLACK;
        const Bitboard white = position.pieces(whiteSource, pieceTypes[i]);
        const Bitboard black = position.pieces(blackSource, pieceTypes[i]);
        out.scalarPlanes[i] = static_cast<std::int8_t>(pieceCount(white) - pieceCount(black));
    }
    out.scalarPlanes[6] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));
    const Color ownColor = position.side_to_move();
    for (const int i : range(5)) {
        out.scalarPlanes[7 + i] = static_cast<std::int8_t>(pieceCount(position.pieces(ownColor, pieceTypes[i])));
    }

    return out;
}

Lc0EncodedBoard encodeLc0Board(const Board &board) {
    Lc0EncodedBoard out{};

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
            ch += Lc0InputDimensions::planesPerHistoryPosition;
        }
    }

    const Color ownColor = position.side_to_move();
    const Color opponentColor = ~ownColor;
    const auto castlingPlane = [&](const Color color, const CastlingRights side) {
        return canonicalBits(allSquares * (position.can_castle(color & side) != 0));
    };
    out.binaryPlanes[ch++] = castlingPlane(ownColor, QUEEN_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(ownColor, KING_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(opponentColor, QUEEN_SIDE);
    out.binaryPlanes[ch++] = castlingPlane(opponentColor, KING_SIDE);
    out.binaryPlanes[ch++] =
        BitBoard<ChessRepresentationDimensions::boardLength>({flipForBlack ? allSquares : 0ULL});

    assert(ch == Lc0InputDimensions::binaryChannelCount);

    // The wrapper module divides this by 99; it is kept as a raw count because the input tensor
    // reaching the network is int8.
    out.scalarPlanes[0] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));
    out.scalarPlanes[1] = 0;
    out.scalarPlanes[2] = 1;

    return out;
}

CompressedEncodedBoard encodeBoard(const Board &board) {
#ifdef CHESS_LC0_INPUT
    return encodeLc0Board(board);
#else
    return encodeProjectBoard(board);
#endif
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
