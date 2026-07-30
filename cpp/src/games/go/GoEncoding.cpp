#include "games/go/GoEncoding.hpp"

#include <limits>
#include <stdexcept>

namespace az::games::go {
namespace {

[[nodiscard]] int32 checkedPointCount(int32 boardSize) {
    if (boardSize < 3) {
        throw std::invalid_argument("Go board size must be at least 3");
    }
    const int64 pointCount = static_cast<int64>(boardSize) * boardSize;
    if (pointCount >= std::numeric_limits<int32>::max()) {
        throw std::invalid_argument("Go board area and pass action must fit in int32");
    }
    return static_cast<int32>(pointCount);
}

} // namespace

std::size_t GoEncodingShape::index(int32 plane, int32 row, int32 column) const {
    if (plane < 0 || plane >= planes || row < 0 || row >= boardSize || column < 0 ||
        column >= boardSize) {
        throw std::out_of_range("Go encoding coordinate out of range");
    }
    return static_cast<std::size_t>(plane) * planeSize +
           static_cast<std::size_t>(row) * static_cast<std::size_t>(boardSize) +
           static_cast<std::size_t>(column);
}

GoEncodingShape checkedEncodingShape(int32 planes, int32 boardSize) {
    if (planes < 1) {
        throw std::invalid_argument("Go encoding plane count must be positive");
    }
    const auto planeSize = static_cast<std::size_t>(checkedPointCount(boardSize));
    const auto planeCount = static_cast<std::size_t>(planes);
    if (planeSize > std::numeric_limits<std::size_t>::max() / planeCount) {
        throw std::length_error("Go encoding size is not representable");
    }
    return GoEncodingShape{
        .planes = planes,
        .boardSize = boardSize,
        .planeSize = planeSize,
        .totalSize = planeCount * planeSize,
    };
}

int8 GoEncoding::at(int32 plane, int32 row, int32 column) const {
    const GoEncodingShape shape = checkedEncodingShape(planes, boardSize);
    if (values.size() != shape.totalSize) {
        throw std::invalid_argument("Go encoding shape is inconsistent");
    }
    return values[shape.index(plane, row, column)];
}

GoEncoding canonicalEncoding(const GoState &state) {
    const int32 historyLength = state.rules().historyLength;
    const int32 boardSize = state.boardSize();
    const int32 planeCount = historyLength * 2 + 1;
    const GoEncodingShape shape = checkedEncodingShape(planeCount, boardSize);
    GoEncoding encoding{
        .planes = planeCount,
        .boardSize = boardSize,
        .values = std::vector<int8>(shape.totalSize, 0),
    };
    const Stone own = state.currentPlayer() == Player::Black ? Stone::Black : Stone::White;
    const Stone opponent = own == Stone::Black ? Stone::White : Stone::Black;
    const auto &history = state.positionHistory();
    for (int32 offset = 0; offset < historyLength; ++offset) {
        if (static_cast<std::size_t>(offset) >= history.size()) {
            break;
        }
        const auto &board = history[history.size() - 1U - static_cast<std::size_t>(offset)];
        for (std::size_t point = 0; point < shape.planeSize; ++point) {
            const Stone stone = board[point];
            if (stone == own) {
                const auto plane = static_cast<std::size_t>(offset) * 2U;
                encoding.values[plane * shape.planeSize + point] = 1;
            } else if (stone == opponent) {
                const auto plane = static_cast<std::size_t>(offset) * 2U + 1U;
                encoding.values[plane * shape.planeSize + point] = 1;
            }
        }
    }
    if (state.currentPlayer() == Player::Black) {
        const auto colorPlane = static_cast<std::size_t>(planeCount - 1);
        for (std::size_t point = 0; point < shape.planeSize; ++point) {
            encoding.values[colorPlane * shape.planeSize + point] = 1;
        }
    }
    return encoding;
}

GoEncoding GoState::canonicalEncoding() const { return go::canonicalEncoding(*this); }

} // namespace az::games::go
