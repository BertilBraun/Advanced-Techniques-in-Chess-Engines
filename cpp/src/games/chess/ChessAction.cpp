#include "games/chess/ChessAction.hpp"
#include "util/py.hpp"

#include "games/chess/ChessBoard.hpp"

#include <array>
#include <cassert>
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
               ChessAction::action_count>;

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
} // namespace

int ChessActionCodec::encode(const ChessAction action, const Board &position) {
    Stockfish::Square fromSquare = action.move.from_sq();
    Stockfish::Square toSquare = action.move.to_sq();
    const Stockfish::PieceType promotionType = action.move.type_of() == Stockfish::PROMOTION
                                                   ? action.move.promotion_type()
                                                   : Stockfish::PieceType::NO_PIECE_TYPE;
    if (position.currentPlayer() == -1) {
        fromSquare = Stockfish::flip_rank(fromSquare);
        toSquare = Stockfish::flip_rank(toSquare);
    }
    const int actionId = moveMappings[fromSquare][toSquare][promotionType];
    assert(0 <= actionId && actionId < ChessAction::action_count);
    return actionId;
}

std::vector<ChessAction> ChessActionCodec::decode(const std::vector<int> &actionIds,
                                                  const Board &position) {
    const std::vector<Stockfish::Move> &legalMoves = position.validMoves();
    std::vector<ChessAction> actions;
    actions.reserve(actionIds.size());
    for (const int actionId : actionIds) {
        assert(0 <= actionId && actionId < ChessAction::action_count);
        auto [fromSquare, toSquare, promotionType] = reverseMoveMappings[actionId];
        if (position.currentPlayer() == -1) {
            fromSquare = Stockfish::flip_rank(fromSquare);
            toSquare = Stockfish::flip_rank(toSquare);
        }
        const auto legal = std::ranges::find_if(legalMoves, [&](const Stockfish::Move move) {
            const bool correctSquares = move.from_sq() == fromSquare && move.to_sq() == toSquare;
            const bool correctPromotion =
                move.type_of() == Stockfish::PROMOTION
                    ? move.promotion_type() == promotionType
                    : promotionType == Stockfish::PieceType::NO_PIECE_TYPE;
            return correctSquares && correctPromotion;
        });
        assert(legal != legalMoves.end());
        actions.emplace_back(*legal);
    }
    return actions;
}
