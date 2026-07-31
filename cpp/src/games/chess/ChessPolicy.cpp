#include "games/chess/ChessPolicy.hpp"

#include <algorithm>
#include <array>
#include <movegen.h>

namespace az::games::chess {
namespace {

constexpr int32 BOARD_LENGTH = 8;
constexpr int32 BOARD_SQUARES = 64;
constexpr int32 PIECE_TYPE_COUNT = 7;

struct MoveDescriptor {
    Stockfish::Square from;
    Stockfish::Square to;
    Stockfish::PieceType promotion;
};

struct PolicyMappings {
    std::array<int32, BOARD_SQUARES * BOARD_SQUARES * PIECE_TYPE_COUNT> forward;
    std::array<MoveDescriptor, CHESS_ACTION_COUNT> reverse;
};

[[nodiscard]] constexpr int32 mappingOffset(int32 from, int32 to, int32 promotion) {
    return (from * BOARD_SQUARES + to) * PIECE_TYPE_COUNT + promotion;
}

void addMapping(PolicyMappings &mappings, int32 &index, int32 from, int32 to,
                Stockfish::PieceType promotion) {
    mappings
        .forward[static_cast<std::size_t>(mappingOffset(from, to, static_cast<int32>(promotion)))] =
        index;
    mappings.reverse[static_cast<std::size_t>(index)] = MoveDescriptor{
        .from = static_cast<Stockfish::Square>(from),
        .to = static_cast<Stockfish::Square>(to),
        .promotion = promotion,
    };
    ++index;
}

[[nodiscard]] PolicyMappings createMappings() {
    PolicyMappings mappings;
    mappings.forward.fill(-1);
    int32 index = 0;
    constexpr std::array<std::pair<int32, int32>, 8> directions{
        {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
    constexpr std::array<std::pair<int32, int32>, 8> knightMoves{
        {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}};
    constexpr std::array<Stockfish::PieceType, 4> promotions{Stockfish::QUEEN, Stockfish::ROOK,
                                                             Stockfish::BISHOP, Stockfish::KNIGHT};

    for (int32 from = 0; from < BOARD_SQUARES; ++from) {
        const int32 row = from / BOARD_LENGTH;
        const int32 column = from % BOARD_LENGTH;
        for (const auto &[rowStep, columnStep] : directions) {
            for (int32 distance = 1; distance < BOARD_LENGTH; ++distance) {
                const int32 toRow = row + rowStep * distance;
                const int32 toColumn = column + columnStep * distance;
                if (toRow >= 0 && toRow < BOARD_LENGTH && toColumn >= 0 &&
                    toColumn < BOARD_LENGTH) {
                    addMapping(mappings, index, from, toRow * BOARD_LENGTH + toColumn,
                               Stockfish::NO_PIECE_TYPE);
                }
            }
        }
        for (const auto &[rowStep, columnStep] : knightMoves) {
            const int32 toRow = row + rowStep;
            const int32 toColumn = column + columnStep;
            if (toRow >= 0 && toRow < BOARD_LENGTH && toColumn >= 0 && toColumn < BOARD_LENGTH) {
                addMapping(mappings, index, from, toRow * BOARD_LENGTH + toColumn,
                           Stockfish::NO_PIECE_TYPE);
            }
        }
        if (row == 6) {
            for (const int32 columnOffset : {-1, 0, 1}) {
                const int32 toColumn = column + columnOffset;
                if (toColumn >= 0 && toColumn < BOARD_LENGTH) {
                    for (const Stockfish::PieceType promotion : promotions) {
                        addMapping(mappings, index, from, 7 * BOARD_LENGTH + toColumn, promotion);
                    }
                }
            }
        }
    }
    assert(index == CHESS_ACTION_COUNT);
    return mappings;
}

[[nodiscard]] const PolicyMappings &policyMappings() {
    static const PolicyMappings mappings = createMappings();
    return mappings;
}

[[nodiscard]] Stockfish::Square canonicalSquare(Stockfish::Square square, Stockfish::Color player) {
    return player == Stockfish::BLACK ? Stockfish::flip_rank(square) : square;
}

} // namespace

int32 encodeMove(Stockfish::Move move, Stockfish::Color player) {
    const Stockfish::Square from = canonicalSquare(move.from_sq(), player);
    const Stockfish::Square to = canonicalSquare(move.to_sq(), player);
    const Stockfish::PieceType promotion =
        move.type_of() == Stockfish::PROMOTION ? move.promotion_type() : Stockfish::NO_PIECE_TYPE;
    const int32 action = policyMappings().forward[static_cast<std::size_t>(
        mappingOffset(from, to, static_cast<int32>(promotion)))];
    if (action < 0 || action >= CHESS_ACTION_COUNT) {
        throw std::logic_error("legal chess move is absent from the policy map");
    }
    return action;
}

std::optional<Stockfish::Move> decodeLegalMove(int32 action, const Stockfish::Position &position) {
    if (action < 0 || action >= CHESS_ACTION_COUNT) {
        return std::nullopt;
    }
    MoveDescriptor descriptor = policyMappings().reverse[static_cast<std::size_t>(action)];
    if (position.side_to_move() == Stockfish::BLACK) {
        descriptor.from = Stockfish::flip_rank(descriptor.from);
        descriptor.to = Stockfish::flip_rank(descriptor.to);
    }
    for (const Stockfish::Move move : Stockfish::MoveList<Stockfish::LEGAL>(position)) {
        const Stockfish::PieceType promotion = move.type_of() == Stockfish::PROMOTION
                                                   ? move.promotion_type()
                                                   : Stockfish::NO_PIECE_TYPE;
        if (move.from_sq() == descriptor.from && move.to_sq() == descriptor.to &&
            promotion == descriptor.promotion) {
            return move;
        }
    }
    return std::nullopt;
}

} // namespace az::games::chess
