#include "games/chess/encoding/ChessEncoding.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <optional>
#include <ranges>
#include <stdexcept>
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
constexpr int promotionRow = 6;

struct PolicyMove {
    Stockfish::Square from;
    Stockfish::Square to;
    Stockfish::PieceType promotion;
};

[[nodiscard]] constexpr std::pair<int, int> squareCoordinates(const int square) {
    return {square / boardLength, square % boardLength};
}

[[nodiscard]] Stockfish::Square square(const int column, const int row) {
    assert(column >= 0 && column < boardLength);
    assert(row >= 0 && row < boardLength);
    return static_cast<Stockfish::Square>(row * boardLength + column);
}

namespace policyPlaneEncoding {
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

[[nodiscard]] int reflectedPlane(const int plane) {
    if (plane < 56) {
        const int directionIndex = plane / 7;
        const int distanceIndex = plane % 7;
        const auto [rowStep, columnStep] = directions[directionIndex];
        const auto reflected = std::ranges::find(directions, std::pair{rowStep, -columnStep});
        assert(reflected != directions.end());
        return static_cast<int>(reflected - directions.begin()) * 7 + distanceIndex;
    }
    if (plane < 64) {
        const auto [rowStep, columnStep] = knightMoves[plane - 56];
        const auto reflected = std::ranges::find(knightMoves, std::pair{rowStep, -columnStep});
        assert(reflected != knightMoves.end());
        return 56 + static_cast<int>(reflected - knightMoves.begin());
    }
    const int promotionIndex = plane - 64;
    const int columnDelta = promotionIndex / static_cast<int>(promotionPieces.size()) - 1;
    const int pieceIndex = promotionIndex % static_cast<int>(promotionPieces.size());
    return 64 + (-columnDelta + 1) * static_cast<int>(promotionPieces.size()) + pieceIndex;
}

[[nodiscard]] int actionId(const Stockfish::Square fromSquare, const Stockfish::Square toSquare,
                           const Stockfish::PieceType promotionType) {
    return policyPlane(fromSquare, toSquare, promotionType) * boardSquareCount +
           static_cast<int>(fromSquare);
}

[[nodiscard]] std::optional<PolicyMove> decode(const int actionId) {
    const int plane = actionId / boardSquareCount;
    const int fromSquare = actionId % boardSquareCount;
    const auto [fromRow, fromColumn] = squareCoordinates(fromSquare);
    int toRow = 0;
    int toColumn = 0;
    Stockfish::PieceType promotion = Stockfish::PieceType::NO_PIECE_TYPE;
    if (plane < 56) {
        const auto [rowStep, columnStep] = directions[plane / 7];
        const int distance = plane % 7 + 1;
        toRow = fromRow + rowStep * distance;
        toColumn = fromColumn + columnStep * distance;
    } else if (plane < 64) {
        const auto [rowStep, columnStep] = knightMoves[plane - 56];
        toRow = fromRow + rowStep;
        toColumn = fromColumn + columnStep;
    } else {
        const int promotionIndex = plane - 64;
        toRow = fromRow + 1;
        toColumn = fromColumn + promotionIndex / static_cast<int>(promotionPieces.size()) - 1;
        promotion = promotionPieces[promotionIndex % static_cast<int>(promotionPieces.size())];
    }
    if (toRow < 0 || toRow >= boardLength || toColumn < 0 || toColumn >= boardLength) {
        return std::nullopt;
    }
    return PolicyMove{
        .from = static_cast<Stockfish::Square>(fromSquare),
        .to = square(toColumn, toRow),
        .promotion = promotion,
    };
}

[[nodiscard]] int mirrorActionId(const int actionId) {
    const int plane = actionId / boardSquareCount;
    const int fromSquare = actionId % boardSquareCount;
    const auto [fromRow, fromColumn] = squareCoordinates(fromSquare);
    return reflectedPlane(plane) * boardSquareCount + square(boardLength - 1 - fromColumn, fromRow);
}
} // namespace policyPlaneEncoding

namespace reducedEncoding {
using MoveMapping =
    std::array<std::array<std::array<int, promotionTypeCount>, boardSquareCount>, boardSquareCount>;
using ReverseMoveMapping =
    std::array<PolicyMove, ChessRepresentationDimensions::reducedActionCount>;

[[nodiscard]] MoveMapping calculateMoveMappings() {
    MoveMapping mappings{};
    for (auto &fromSquare : mappings) {
        for (auto &toSquare : fromSquare) {
            toSquare.fill(-1);
        }
    }

    int nextActionId = 0;
    const auto addMove = [&mappings, &nextActionId](const int fromSquare, const int toSquare,
                                                    const Stockfish::PieceType promotionType) {
        mappings[fromSquare][toSquare][static_cast<int>(promotionType)] = nextActionId++;
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
        if (row == promotionRow) {
            for (const int columnOffset : {-1, 0, 1}) {
                const int toColumn = column + columnOffset;
                if (0 <= toColumn && toColumn < boardLength) {
                    const int toSquare = square(toColumn, promotionRow + 1);
                    for (const Stockfish::PieceType promotionType : promotionPieces) {
                        addMove(fromSquare, toSquare, promotionType);
                    }
                }
            }
        }
    }
    assert(nextActionId == ChessRepresentationDimensions::reducedActionCount);
    return mappings;
}

[[nodiscard]] ReverseMoveMapping calculateReverseMoveMappings(const MoveMapping &mappings) {
    ReverseMoveMapping reverseMappings{};
    for (const int fromSquare : range(boardSquareCount)) {
        for (const int toSquare : range(boardSquareCount)) {
            for (const int promotionType : range(promotionTypeCount)) {
                const int actionId = mappings[fromSquare][toSquare][promotionType];
                if (actionId >= 0) {
                    reverseMappings[actionId] = PolicyMove{
                        .from = static_cast<Stockfish::Square>(fromSquare),
                        .to = static_cast<Stockfish::Square>(toSquare),
                        .promotion = static_cast<Stockfish::PieceType>(promotionType),
                    };
                }
            }
        }
    }
    return reverseMappings;
}

const MoveMapping moveMappings = calculateMoveMappings();
const ReverseMoveMapping reverseMoveMappings = calculateReverseMoveMappings(moveMappings);

[[nodiscard]] int actionId(const Stockfish::Square fromSquare, const Stockfish::Square toSquare,
                           const Stockfish::PieceType promotionType) {
    return moveMappings[fromSquare][toSquare][static_cast<int>(promotionType)];
}

[[nodiscard]] std::optional<PolicyMove> decode(const int actionId) {
    return reverseMoveMappings[actionId];
}

[[nodiscard]] int mirrorActionId(const int actionId) {
    const PolicyMove move = reverseMoveMappings[actionId];
    const auto [fromRow, fromColumn] = squareCoordinates(move.from);
    const auto [toRow, toColumn] = squareCoordinates(move.to);
    const int mirroredFrom = square(boardLength - 1 - fromColumn, fromRow);
    const int mirroredTo = square(boardLength - 1 - toColumn, toRow);
    return moveMappings[mirroredFrom][mirroredTo][static_cast<int>(move.promotion)];
}
} // namespace reducedEncoding

[[nodiscard]] int encodedActionId(const Stockfish::Square fromSquare,
                                  const Stockfish::Square toSquare,
                                  const Stockfish::PieceType promotionType) {
    if constexpr (chessActionEncoding == ChessActionEncoding::Reduced) {
        return reducedEncoding::actionId(fromSquare, toSquare, promotionType);
    } else {
        return policyPlaneEncoding::actionId(fromSquare, toSquare, promotionType);
    }
}

[[nodiscard]] std::optional<PolicyMove> decodeCanonicalActionId(const int actionId) {
    if constexpr (chessActionEncoding == ChessActionEncoding::Reduced) {
        return reducedEncoding::decode(actionId);
    } else {
        return policyPlaneEncoding::decode(actionId);
    }
}
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
    const int actionId = encodedActionId(fromSquare, toSquare, promotionType);
    assert(0 <= actionId && actionId < ChessEncoding::actionCount);
    return actionId;
}

ChessAction ChessEncoding::decodeAction(const int actionId, const Board &state) {
    if (actionId < 0 || actionId >= actionCount) {
        throw std::invalid_argument("Chess action ID is outside the action space");
    }
    const std::optional<PolicyMove> decoded = decodeCanonicalActionId(actionId);
    if (!decoded.has_value()) {
        throw std::invalid_argument("Chess action ID encodes an off-board move");
    }
    Stockfish::Square fromSquare = decoded->from;
    Stockfish::Square toSquare = decoded->to;
    if (state.currentPlayer() == -1) {
        fromSquare = Stockfish::flip_rank(fromSquare);
        toSquare = Stockfish::flip_rank(toSquare);
    }
    const std::vector<Stockfish::Move> &legalMoves = state.validMoves();
    const auto legal = std::ranges::find_if(legalMoves, [&](const Stockfish::Move move) {
        const bool correctSquares = move.from_sq() == fromSquare && move.to_sq() == toSquare;
        const bool correctPromotion =
            move.type_of() == Stockfish::PROMOTION
                ? move.promotion_type() == decoded->promotion
                : decoded->promotion == Stockfish::PieceType::NO_PIECE_TYPE;
        return correctSquares && correctPromotion;
    });
    if (legal == legalMoves.end()) {
        throw std::invalid_argument("Chess action ID is illegal in this position");
    }
    return ChessAction(*legal);
}

int ChessEncoding::mirrorActionId(const int actionId) {
    assert(0 <= actionId && actionId < actionCount);
    int mirrored = 0;
    if constexpr (chessActionEncoding == ChessActionEncoding::Reduced) {
        mirrored = reducedEncoding::mirrorActionId(actionId);
    } else {
        mirrored = policyPlaneEncoding::mirrorActionId(actionId);
    }
    assert(0 <= mirrored && mirrored < actionCount);
    return mirrored;
}
