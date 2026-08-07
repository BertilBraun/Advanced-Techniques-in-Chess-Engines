#include "games/chess/encoding/ChessEncoding.hpp"
#include "util/py.hpp"

#include "util/TimeItGuard.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
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
    for (const BitBoard<ChessRepresentationDimensions::board_length> &board : compressed.bits) {
        for (const std::uint64_t value : board.words()) {
            first = splitmix64(first ^ value);
            second = splitmix64(second ^ std::rotl(value, 31));
        }
    }
    for (const std::int8_t value : compressed.scal) {
        const std::uint64_t unsignedValue = static_cast<std::uint8_t>(value);
        first = splitmix64(first ^ unsignedValue);
        second = splitmix64(second ^ (unsignedValue + 0x9E3779B97F4A7C15ULL));
    }
    return {.first = first, .second = second};
}

CompressedEncodedBoard encodeBoard(const Board &board) {
    TIMEIT("encodeBoard");
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
            out.bits[ch++] = canonicalBits(position.pieces(positionColor, piece));
        }
    }

    constexpr std::uint64_t ALL_SET = 0xFFFF'FFFF'FFFF'FFFFull;
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.bits[ch++] = canonicalBits(ALL_SET * position.can_castle(positionColor & KING_SIDE));
        out.bits[ch++] = canonicalBits(ALL_SET * position.can_castle(positionColor & QUEEN_SIDE));
    }

    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        out.bits[ch++] = canonicalBits(position.pieces(positionColor));
    }

    out.bits[ch++] = canonicalBits(position.checkers());

    const Square epSquare = position.ep_square();
    out.bits[ch++] = canonicalBits(epSquare == SQ_NONE ? 0ULL : square_bb(epSquare));
    out.bits[ch++] = canonicalBits(ALL_SET * (board.repetitionCount() >= 1));
    out.bits[ch++] = canonicalBits(ALL_SET * (board.repetitionCount() >= 2));

    assert(ch == ChessRepresentationDimensions::binary_channel_count);

    for (const int i : range(6)) {
        const Color whiteSource = flipForBlack ? BLACK : WHITE;
        const Color blackSource = flipForBlack ? WHITE : BLACK;
        const Bitboard white = position.pieces(whiteSource, pieceTypes[i]);
        const Bitboard black = position.pieces(blackSource, pieceTypes[i]);
        out.scal[i] = static_cast<std::int8_t>(pieceCount(white) - pieceCount(black));
    }
    out.scal[6] = static_cast<std::int8_t>(std::min(position.rule50_count(), 100));

    return out;
}

torch::Tensor tensorEncoding(const CompressedEncodedBoard &compressed) {
    TIMEIT("toTensor");

    auto tensor = torch::empty({ChessRepresentationDimensions::channel_count,
                                ChessRepresentationDimensions::board_length,
                                ChessRepresentationDimensions::board_length},
                               torch::TensorOptions().dtype(torch::kInt8));

    writeTensorEncoding(compressed, tensor.data_ptr<std::int8_t>());
    return tensor;
}

void writePackedPlaneEncoding(const CompressedEncodedBoard &compressed, std::int8_t *destination) {
    assert(destination != nullptr);
    serialize_binary_planes<ChessRepresentationDimensions::board_length,
                            ChessRepresentationDimensions::binary_channel_count>(
        compressed.bits,
        std::span<std::int8_t>(destination, CompressedEncodedBoard::packed_binary_bytes));
    std::memcpy(destination + CompressedEncodedBoard::packed_binary_bytes, compressed.scal.data(),
                compressed.scal.size() * sizeof(std::int8_t));
}

void writeTensorEncoding(const CompressedEncodedBoard &compressed, std::int8_t *destination) {
    assert(destination != nullptr);

    for (const auto channelIndex :
         range(static_cast<std::size_t>(ChessRepresentationDimensions::binary_channel_count))) {
        const std::uint64_t bits = compressed.bits[channelIndex].word(0);
        std::int8_t *planeDestination =
            destination + static_cast<std::ptrdiff_t>(channelIndex) * 64;

        for (const int byte : range(8)) {
            const std::uint8_t value = (bits >> (byte * 8)) & 0xFFu;
            planeDestination[byte * 8 + 0] = value & 1;
            planeDestination[byte * 8 + 1] = (value >> 1) & 1;
            planeDestination[byte * 8 + 2] = (value >> 2) & 1;
            planeDestination[byte * 8 + 3] = (value >> 3) & 1;
            planeDestination[byte * 8 + 4] = (value >> 4) & 1;
            planeDestination[byte * 8 + 5] = (value >> 5) & 1;
            planeDestination[byte * 8 + 6] = (value >> 6) & 1;
            planeDestination[byte * 8 + 7] = (value >> 7) & 1;
        }
    }

    for (const auto scalarIndex :
         range(static_cast<std::size_t>(ChessRepresentationDimensions::scalar_channel_count))) {
        std::int8_t *scalarDestination =
            destination + static_cast<std::ptrdiff_t>(
                              ChessRepresentationDimensions::binary_channel_count + scalarIndex) *
                              64;
        std::memset(scalarDestination, compressed.scal[scalarIndex], 64);
    }
}

void encodeBoardInto(const Board &board, std::int8_t *destination) {
    writeTensorEncoding(encodeBoard(board), destination);
}

void ChessEncoding::encodeInputInto(const Board &state, std::int8_t *destination) {
    encodeBoardInto(state, destination);
}
