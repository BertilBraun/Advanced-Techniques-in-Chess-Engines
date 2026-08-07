#include "games/chess/ChessAction.hpp"

#include "common.hpp"

#include <array>
#include <cassert>
#include <tuple>
#include <utility>
#include <vector>

namespace {
constexpr std::array<std::pair<int, int>, 8> directions = {
    {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
constexpr std::array<std::pair<int, int>, 8> knightMoves = {
    {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}};
constexpr std::array promotionPieces = {
    PieceType::QUEEN,
    PieceType::ROOK,
    PieceType::BISHOP,
    PieceType::KNIGHT,
};
constexpr int promotionTypeCount = static_cast<int>(PieceType::PIECE_TYPE_NB);

using MoveMapping =
    std::array<std::array<std::array<int, promotionTypeCount>, BOARD_SIZE>, BOARD_SIZE>;

MoveMapping calculateMoveMappings() {
    MoveMapping mappings{};
    for (auto &fromSquare : mappings) {
        for (auto &toSquare : fromSquare) {
            toSquare.fill(-1);
        }
    }

    int actionId = 0;
    const auto addMove = [&mappings, &actionId](const int fromSquare, const int toSquare,
                                                const PieceType promotionType) {
        mappings[fromSquare][toSquare][static_cast<int>(promotionType)] = actionId++;
    };
    for (int fromSquare = 0; fromSquare < BOARD_SIZE; ++fromSquare) {
        const auto [row, column] = squareToIndex(fromSquare);
        for (const auto &[rowStep, columnStep] : directions) {
            for (int distance = 1; distance < BOARD_LENGTH; ++distance) {
                const int toRow = row + rowStep * distance;
                const int toColumn = column + columnStep * distance;
                if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toColumn &&
                    toColumn < BOARD_LENGTH) {
                    addMove(fromSquare, square(toColumn, toRow), PieceType::NO_PIECE_TYPE);
                }
            }
        }
        for (const auto &[rowStep, columnStep] : knightMoves) {
            const int toRow = row + rowStep;
            const int toColumn = column + columnStep;
            if (0 <= toRow && toRow < BOARD_LENGTH && 0 <= toColumn && toColumn < BOARD_LENGTH) {
                addMove(fromSquare, square(toColumn, toRow), PieceType::NO_PIECE_TYPE);
            }
        }
        if (row == 6) {
            for (const int columnOffset : {-1, 0, 1}) {
                if (0 <= column + columnOffset && column + columnOffset < BOARD_LENGTH) {
                    const int toSquare = square(column + columnOffset, 7);
                    for (const PieceType promotionType : promotionPieces) {
                        addMove(fromSquare, toSquare, promotionType);
                    }
                }
            }
        }
    }
    return mappings;
}

using ReverseMoveMapping = std::array<std::tuple<Square, Square, PieceType>, ACTION_SIZE>;

ReverseMoveMapping calculateReverseMoveMappings(const MoveMapping &mappings) {
    ReverseMoveMapping reverseMappings{};
    for (int fromSquare = 0; fromSquare < BOARD_SIZE; ++fromSquare) {
        for (int toSquare = 0; toSquare < BOARD_SIZE; ++toSquare) {
            for (int promotionType = 0; promotionType < promotionTypeCount; ++promotionType) {
                const int actionId = mappings[fromSquare][toSquare][promotionType];
                if (actionId >= 0) {
                    reverseMappings[actionId] = {static_cast<Square>(fromSquare),
                                                 static_cast<Square>(toSquare),
                                                 static_cast<PieceType>(promotionType)};
                }
            }
        }
    }
    return reverseMappings;
}

const MoveMapping moveMappings = calculateMoveMappings();
const ReverseMoveMapping reverseMoveMappings = calculateReverseMoveMappings(moveMappings);
} // namespace

int ChessActionCodec::encode(const ChessAction action, const Board &position) {
    Square fromSquare = action.move.from_sq();
    Square toSquare = action.move.to_sq();
    const PieceType promotionType = action.move.type_of() == PROMOTION
                                        ? action.move.promotion_type()
                                        : PieceType::NO_PIECE_TYPE;
    if (position.currentPlayer() == -1) {
        fromSquare = flip_rank(fromSquare);
        toSquare = flip_rank(toSquare);
    }
    const int actionId = moveMappings[fromSquare][toSquare][promotionType];
    assert(0 <= actionId && actionId < ACTION_SIZE);
    return actionId;
}

std::vector<ChessAction> ChessActionCodec::decode(const std::vector<int> &actionIds,
                                                  const Board &position) {
    const std::vector<Move> &legalMoves = position.validMoves();
    std::vector<ChessAction> actions;
    actions.reserve(actionIds.size());
    for (const int actionId : actionIds) {
        assert(0 <= actionId && actionId < ACTION_SIZE);
        auto [fromSquare, toSquare, promotionType] = reverseMoveMappings[actionId];
        if (position.currentPlayer() == -1) {
            fromSquare = flip_rank(fromSquare);
            toSquare = flip_rank(toSquare);
        }
        const auto legal = std::ranges::find_if(legalMoves, [&](const Move move) {
            const bool correctSquares = move.from_sq() == fromSquare && move.to_sq() == toSquare;
            const bool correctPromotion = move.type_of() == PROMOTION
                                              ? move.promotion_type() == promotionType
                                              : promotionType == PieceType::NO_PIECE_TYPE;
            return correctSquares && correctPromotion;
        });
        assert(legal != legalMoves.end());
        actions.emplace_back(*legal);
    }
    return actions;
}
