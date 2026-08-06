#include "BoardEncoding.hpp"

#include <span>

constexpr int pieceCount(const Bitboard bb) noexcept { return std::popcount(bb); }

constexpr Bitboard flipRanks(const Bitboard bits) noexcept {
    return ((bits & 0x0000'0000'0000'00FFULL) << 56) | ((bits & 0x0000'0000'0000'FF00ULL) << 40) |
           ((bits & 0x0000'0000'00FF'0000ULL) << 24) | ((bits & 0x0000'0000'FF00'0000ULL) << 8) |
           ((bits & 0x0000'00FF'0000'0000ULL) >> 8) | ((bits & 0x0000'FF00'0000'0000ULL) >> 24) |
           ((bits & 0x00FF'0000'0000'0000ULL) >> 40) | ((bits & 0xFF00'0000'0000'0000ULL) >> 56);
}

constexpr uint64 splitmix64(uint64 x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::size_t BoardFingerprintHash::operator()(const BoardFingerprint &fingerprint) const noexcept {
    return static_cast<std::size_t>(fingerprint.first ^ std::rotl(fingerprint.second, 29));
}

BoardFingerprint fingerprintBoard(const CompressedEncodedBoard &compressed) {
    uint64 first = 0x243F6A8885A308D3ULL;
    uint64 second = 0x13198A2E03707344ULL;
    for (const BitBoard<BOARD_LEN> &board : compressed.bits) {
        for (const uint64 value : board.words()) {
            first = splitmix64(first ^ value);
            second = splitmix64(second ^ std::rotl(value, 31));
        }
    }
    for (const int8 value : compressed.scal) {
        const uint64 unsignedValue = static_cast<uint8>(value);
        first = splitmix64(first ^ unsignedValue);
        second = splitmix64(second ^ (unsignedValue + 0x9E3779B97F4A7C15ULL));
    }
    return {first, second};
}

CompressedEncodedBoard encodeBoard(const Board *board) {
    TIMEIT("encodeBoard");
    CompressedEncodedBoard out{};

    // Canonical chess inputs are always encoded from the side-to-move perspective.
    const Position &position = board->position();
    const bool flipForBlack = position.side_to_move() == BLACK;
    const auto canonicalBits = [flipForBlack](const Bitboard bits) {
        return BitBoard<BOARD_LEN>({flipForBlack ? flipRanks(bits) : bits});
    };

    int ch = 0;
    // Piece-type planes remain Stockfish bitboards until the encoding boundary.
    for (const Color color : {WHITE, BLACK}) {
        const Color positionColor = flipForBlack ? ~color : color;
        for (const PieceType piece : PIECE_TYPES) {
            out.bits[ch++] = canonicalBits(position.pieces(positionColor, piece));
        }
    }

    constexpr uint64 ALL_SET = 0xFFFF'FFFF'FFFF'FFFFull;
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
    out.bits[ch++] = canonicalBits(ALL_SET * (board->repetitionCount() >= 1));
    out.bits[ch++] = canonicalBits(ALL_SET * (board->repetitionCount() >= 2));

    assert(ch == BINARY_C);

    for (int i = 0; i < 6; ++i) {
        const Color whiteSource = flipForBlack ? BLACK : WHITE;
        const Color blackSource = flipForBlack ? WHITE : BLACK;
        const Bitboard white = position.pieces(whiteSource, PIECE_TYPES[i]);
        const Bitboard black = position.pieces(blackSource, PIECE_TYPES[i]);
        out.scal[i] = static_cast<int8>(pieceCount(white) - pieceCount(black));
    }
    out.scal[6] = static_cast<int8>(std::min(position.rule50_count(), 100));

    return out;
}

torch::Tensor toTensor(const CompressedEncodedBoard &compressed) {
    TIMEIT("toTensor");

    auto tensor =
        torch::empty({BOARD_C, BOARD_LEN, BOARD_LEN}, torch::TensorOptions().dtype(torch::kInt8));

    writeTensorEncoding(compressed, tensor.data_ptr<int8>());
    return tensor;
}

void writePackedPlaneEncoding(const CompressedEncodedBoard &compressed, int8 *destination) {
    assert(destination != nullptr);
    serialize_binary_planes<BOARD_LEN, BINARY_C>(
        compressed.bits,
        std::span<int8>(destination, CHESS_PACKED_BINARY_BYTES));
    std::memcpy(
        destination + CHESS_PACKED_BINARY_BYTES,
        compressed.scal.data(),
        compressed.scal.size() * sizeof(int8));
}

void writeTensorEncoding(const CompressedEncodedBoard &compressed, int8 *destination) {
    assert(destination != nullptr);

    for (std::size_t channelIndex = 0; channelIndex < static_cast<std::size_t>(BINARY_C); ++channelIndex) {
        const uint64 bits = compressed.bits[channelIndex].word(0);
        int8 *planeDestination = destination + static_cast<std::ptrdiff_t>(channelIndex) * 64;

        for (int byte = 0; byte < 8; ++byte) {
            const uint8 value = (bits >> (byte * 8)) & 0xFFu;
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

    for (std::size_t scalarIndex = 0; scalarIndex < static_cast<std::size_t>(SCALAR_C); ++scalarIndex) {
        int8 *scalarDestination =
            destination + static_cast<std::ptrdiff_t>(BINARY_C + scalarIndex) * 64;
        std::memset(scalarDestination, compressed.scal[scalarIndex], 64);
    }
}

void encodeBoardInto(const Board &board, int8 *destination) {
    writeTensorEncoding(encodeBoard(&board), destination);
}

float getBoardResultScore(const Board &board) {
    assert(board.isGameOver() && "Game was not over!");

    if (const auto winner = board.checkWinner()) {
        assert(winner.value() != board.currentPlayer());
        return -1.0f; // The side to move is checkmated.
    }

    return 0.0f;
}
