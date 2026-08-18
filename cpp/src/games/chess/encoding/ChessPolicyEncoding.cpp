#include "games/chess/encoding/ChessEncoding.hpp"
#include "util/py.hpp"

#include "games/chess/implementation/ChessBoard.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

namespace {
constexpr int boardLength = 8;
constexpr int boardSquareCount = boardLength * boardLength;
constexpr std::array<std::pair<int, int>, 8> directions = {
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
constexpr std::array<std::pair<int, int>, 8> knightMoves = {
    {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}};
constexpr std::array promotionPieces = {
    Stockfish::PieceType::QUEEN,
    Stockfish::PieceType::ROOK,
    Stockfish::PieceType::BISHOP,
    Stockfish::PieceType::KNIGHT,
};
constexpr int promotionTypeCount = static_cast<int>(Stockfish::PieceType::PIECE_TYPE_NB);

using MoveMapping =
    std::array<std::array<std::array<int, promotionTypeCount>, boardSquareCount>, boardSquareCount>;
using ReverseMoveMapping =
    std::array<std::tuple<Stockfish::Square, Stockfish::Square, Stockfish::PieceType>,
               ChessEncoding::action_count>;

[[nodiscard]] constexpr std::pair<int, int> squareCoordinates(const int square) {
    return {square / boardLength, square % boardLength};
}

[[nodiscard]] Stockfish::Square square(const int column, const int row) {
    assert(column >= 0 && column < boardLength);
    assert(row >= 0 && row < boardLength);
    return static_cast<Stockfish::Square>(row * boardLength + column);
}

[[nodiscard]] MoveMapping calculateMoveMappings() {
    MoveMapping mappings{};
    for (auto &fromSquare : mappings) {
        for (auto &toSquare : fromSquare) {
            toSquare.fill(-1);
        }
    }

    int actionId = 0;
    const auto addMove = [&mappings, &actionId](const int fromSquare, const int toSquare,
                                                const Stockfish::PieceType promotionType) {
        mappings[fromSquare][toSquare][static_cast<int>(promotionType)] = actionId++;
    };
    for (const int fromSquare : range(boardSquareCount)) {
        const auto [row, column] = squareCoordinates(fromSquare);
        for (const auto &[rowStep, columnStep] : directions) {
            for (const int distance : range(1, boardLength)) {
                const int toRow = row + rowStep * distance;
                const int toColumn = column + columnStep * distance;
                if (0 <= toRow && toRow < boardLength && 0 <= toColumn && toColumn < boardLength) {
                    addMove(fromSquare, square(toColumn, toRow),
                            Stockfish::PieceType::NO_PIECE_TYPE);
                }
            }
        }
        for (const auto &[rowStep, columnStep] : knightMoves) {
            const int toRow = row + rowStep;
            const int toColumn = column + columnStep;
            if (0 <= toRow && toRow < boardLength && 0 <= toColumn && toColumn < boardLength) {
                addMove(fromSquare, square(toColumn, toRow), Stockfish::PieceType::NO_PIECE_TYPE);
            }
        }
        if (row == 6) {
            for (const int columnOffset : {-1, 0, 1}) {
                if (0 <= column + columnOffset && column + columnOffset < boardLength) {
                    const int toSquare = square(column + columnOffset, 7);
                    for (const Stockfish::PieceType promotionType : promotionPieces) {
                        addMove(fromSquare, toSquare, promotionType);
                    }
                }
            }
        }
    }
    return mappings;
}

[[nodiscard]] ReverseMoveMapping calculateReverseMoveMappings(const MoveMapping &mappings) {
    ReverseMoveMapping reverseMappings{};
    for (const int fromSquare : range(boardSquareCount)) {
        for (const int toSquare : range(boardSquareCount)) {
            for (const int promotionType : range(promotionTypeCount)) {
                const int actionId = mappings[fromSquare][toSquare][promotionType];
                if (actionId >= 0) {
                    reverseMappings[actionId] = {static_cast<Stockfish::Square>(fromSquare),
                                                 static_cast<Stockfish::Square>(toSquare),
                                                 static_cast<Stockfish::PieceType>(promotionType)};
                }
            }
        }
    }
    return reverseMappings;
}

const MoveMapping moveMappings = calculateMoveMappings();
const ReverseMoveMapping reverseMoveMappings = calculateReverseMoveMappings(moveMappings);

[[nodiscard]] int policyPlane(const Stockfish::Square fromSquare, const Stockfish::Square toSquare,
                              const Stockfish::PieceType promotionType) {
    const auto [fromRow, fromColumn] = squareCoordinates(fromSquare);
    const auto [toRow, toColumn] = squareCoordinates(toSquare);
    const int rowDelta = toRow - fromRow;
    const int columnDelta = toColumn - fromColumn;
    if (promotionType != Stockfish::PieceType::NO_PIECE_TYPE) {
        const auto piece = std::ranges::find(promotionPieces, promotionType);
        assert(piece != promotionPieces.end());
        return 64 + (columnDelta + 1) * static_cast<int>(promotionPieces.size()) +
               static_cast<int>(piece - promotionPieces.begin());
    }
    const auto knight = std::ranges::find(knightMoves, std::pair{rowDelta, columnDelta});
    if (knight != knightMoves.end()) {
        return 56 + static_cast<int>(knight - knightMoves.begin());
    }
    const int distance = std::max(std::abs(rowDelta), std::abs(columnDelta));
    const std::pair direction{rowDelta / distance, columnDelta / distance};
    const auto ray = std::ranges::find(directions, direction);
    assert(ray != directions.end());
    return static_cast<int>(ray - directions.begin()) * 7 + distance - 1;
}

[[nodiscard]] std::array<int, ChessEncoding::action_count> calculatePolicyPlaneIndices() {
    std::array<int, ChessEncoding::action_count> indices{};
    std::array<bool, ChessEncoding::policy_plane_count * boardSquareCount> occupied{};
    for (const int actionId : range(ChessEncoding::action_count)) {
        const auto [fromSquare, toSquare, promotionType] = reverseMoveMappings[actionId];
        const int index = policyPlane(fromSquare, toSquare, promotionType) * boardSquareCount +
                          static_cast<int>(fromSquare);
        assert(0 <= index && index < static_cast<int>(occupied.size()));
        assert(!occupied[index]);
        occupied[index] = true;
        indices[actionId] = index;
    }
    return indices;
}

const std::array<int, ChessEncoding::action_count> policyPlaneIndices =
    calculatePolicyPlaneIndices();
} // namespace

int ChessEncoding::actionId(const ChessAction action, const Board &state) {
    Stockfish::Square fromSquare = action.move.from_sq();
    Stockfish::Square toSquare = action.move.to_sq();
    const Stockfish::PieceType promotionType = action.move.type_of() == Stockfish::PROMOTION
                                                   ? action.move.promotion_type()
                                                   : Stockfish::PieceType::NO_PIECE_TYPE;
    if (state.currentPlayer() == -1) {
        fromSquare = Stockfish::flip_rank(fromSquare);
        toSquare = Stockfish::flip_rank(toSquare);
    }
    const int actionId = moveMappings[fromSquare][toSquare][promotionType];
    assert(0 <= actionId && actionId < ChessEncoding::action_count);
    return actionId;
}

ChessAction ChessEncoding::decodeAction(const int actionId, const Board &state) {
    assert(0 <= actionId && actionId < action_count);
    const std::vector<Stockfish::Move> &legalMoves = state.validMoves();
    auto [fromSquare, toSquare, promotionType] = reverseMoveMappings[actionId];
    if (state.currentPlayer() == -1) {
        fromSquare = Stockfish::flip_rank(fromSquare);
        toSquare = Stockfish::flip_rank(toSquare);
    }
    const auto legal = std::ranges::find_if(legalMoves, [&](const Stockfish::Move candidate) {
        const bool correctSquares =
            candidate.from_sq() == fromSquare && candidate.to_sq() == toSquare;
        const bool correctPromotion = candidate.type_of() == Stockfish::PROMOTION
                                          ? candidate.promotion_type() == promotionType
                                          : promotionType == Stockfish::PieceType::NO_PIECE_TYPE;
        return correctSquares && correctPromotion;
    });
    assert(legal != legalMoves.end());
    return ChessAction(*legal);
}

int ChessEncoding::mirrorActionId(const int actionId) {
    assert(0 <= actionId && actionId < action_count);
    const auto [fromSquare, toSquare, promotionType] = reverseMoveMappings[actionId];
    const auto [fromRow, fromColumn] = squareCoordinates(fromSquare);
    const auto [toRow, toColumn] = squareCoordinates(toSquare);
    const int mirroredFrom = square(boardLength - 1 - fromColumn, fromRow);
    const int mirroredTo = square(boardLength - 1 - toColumn, toRow);
    const int mirrored = moveMappings[mirroredFrom][mirroredTo][promotionType];
    assert(0 <= mirrored && mirrored < action_count);
    return mirrored;
}

const std::array<int, ChessEncoding::action_count> &ChessEncoding::policyPlaneIndices() {
    return ::policyPlaneIndices;
}
