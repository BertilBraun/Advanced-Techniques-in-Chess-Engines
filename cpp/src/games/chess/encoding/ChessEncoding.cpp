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

constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::size_t BoardFingerprintHash::operator()(const BoardFingerprint &fingerprint) const noexcept {
    return static_cast<std::size_t>(fingerprint.first ^ std::rotl(fingerprint.second, 29));
}

BoardFingerprint fingerprintBoard(const CompressedEncodedBoard &compressed) {
    std::uint64_t first = 0x243F6A8885A308D3ULL;
    std::uint64_t second = 0x13198A2E03707344ULL;
    for (const BitBoard<ChessRepresentationDimensions::board_length> &board :
         compressed.binary_planes) {
        for (const std::uint64_t value : board.words()) {
            first = splitmix64(first ^ value);
            second = splitmix64(second ^ std::rotl(value, 31));
        }
    }
    for (const std::int8_t value : compressed.scalar_planes) {
        const std::uint64_t unsignedValue = static_cast<std::uint8_t>(value);
        first = splitmix64(first ^ unsignedValue);
        second = splitmix64(second ^ (unsignedValue + 0x9E3779B97F4A7C15ULL));
    }
    return {.first = first, .second = second};
}

CompressedEncodedBoard encodeBoard(const Board &board) {
    CompressedEncodedBoard out{};

    // Canonical chess inputs are always encoded from the side-to-move perspective.
    const Position &position = board.position();
    const bool flipForBlack = position.side_to_move() == BLACK;
    const auto canonicalBits = [flipForBlack](const Bitboard bits) {
        return BitBoard<ChessRepresentationDimensions::board_length>(
            {flipForBlack ? flipRanks(bits) : bits});
    };

    int ch = 0;
    // Piece-type planes remain Stockfish bitboards until the encoding boundary.
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        for (const PieceType piece : pieceTypes) {
            out.binary_planes[ch++] = canonicalBits(position.pieces(positionColor, piece));
        }
    }

    constexpr std::uint64_t ALL_SET = 0xFFFF'FFFF'FFFF'FFFFull;
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binary_planes[ch++] =
            canonicalBits(ALL_SET * position.can_castle(positionColor & KING_SIDE));
        out.binary_planes[ch++] =
            canonicalBits(ALL_SET * position.can_castle(positionColor & QUEEN_SIDE));
    }

    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.binary_planes[ch++] = canonicalBits(position.pieces(positionColor));
    }

    out.binary_planes[ch++] = canonicalBits(position.checkers());

    const Square epSquare = position.ep_square();
    out.binary_planes[ch++] = canonicalBits(epSquare == SQ_NONE ? 0ULL : square_bb(epSquare));
    out.binary_planes[ch++] = canonicalBits(ALL_SET * (board.repetitionCount() >= 1));
    out.binary_planes[ch++] = canonicalBits(ALL_SET * (board.repetitionCount() >= 2));

    assert(ch == ChessRepresentationDimensions::binary_channel_count);

    for (const int i : range(6)) {
        const Color whiteSource = flipForBlack ? BLACK : WHITE;
        const Color blackSource = flipForBlack ? WHITE : BLACK;
        const Bitboard white = position.pieces(whiteSource, pieceTypes[i]);
        const Bitboard black = position.pieces(blackSource, pieceTypes[i]);
        out.scalar_planes[i] = static_cast<std::int8_t>(pieceCount(white) - pieceCount(black));
    }
    out.scalar_planes[6] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));

    return out;
}

torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed) {
    auto tensor = torch::empty({ChessRepresentationDimensions::channel_count,
                                ChessRepresentationDimensions::board_length,
                                ChessRepresentationDimensions::board_length},
                               torch::TensorOptions().dtype(torch::kInt8));

    compressed.writeTensorInto(std::span<std::int8_t>(tensor.data_ptr<std::int8_t>(),
                                                      CompressedEncodedBoard::tensor_values));
    return tensor;
}

void ChessEncoding::encodeInputInto(const Board &state, std::int8_t *destination) {
    const CompressedEncodedBoard encoded = encodeBoard(state);
    encoded.writeTensorInto(
        std::span<std::int8_t>(destination, CompressedEncodedBoard::tensor_values));
}
