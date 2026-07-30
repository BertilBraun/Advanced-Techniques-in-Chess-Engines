#include "games/go/GoSymmetry.hpp"

#include <limits>
#include <stdexcept>

namespace az::games::go {
namespace {

int32 checkedPointCount(int32 boardSize) {
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

Coordinate transformCoordinate(int32 row, int32 column, int32 boardSize, Symmetry symmetry) {
    checkedPointCount(boardSize);
    if (row < 0 || row >= boardSize || column < 0 || column >= boardSize) {
        throw std::invalid_argument("Go coordinate out of range");
    }
    const int32 maximum = boardSize - 1;
    switch (symmetry) {
    case Symmetry::Identity:
        return Coordinate{.row = row, .column = column};
    case Symmetry::Rotate90:
        return Coordinate{.row = column, .column = maximum - row};
    case Symmetry::Rotate180:
        return Coordinate{.row = maximum - row, .column = maximum - column};
    case Symmetry::Rotate270:
        return Coordinate{.row = maximum - column, .column = row};
    case Symmetry::Reflect:
        return Coordinate{.row = row, .column = maximum - column};
    case Symmetry::ReflectRotate90:
        return Coordinate{.row = maximum - column, .column = maximum - row};
    case Symmetry::ReflectRotate180:
        return Coordinate{.row = maximum - row, .column = column};
    case Symmetry::ReflectRotate270:
        return Coordinate{.row = column, .column = row};
    }
    throw std::invalid_argument("Unknown Go symmetry");
}

Symmetry inverseSymmetry(Symmetry symmetry) {
    switch (symmetry) {
    case Symmetry::Identity:
        return Symmetry::Identity;
    case Symmetry::Rotate90:
        return Symmetry::Rotate270;
    case Symmetry::Rotate180:
        return Symmetry::Rotate180;
    case Symmetry::Rotate270:
        return Symmetry::Rotate90;
    case Symmetry::Reflect:
        return Symmetry::Reflect;
    case Symmetry::ReflectRotate90:
        return Symmetry::ReflectRotate90;
    case Symmetry::ReflectRotate180:
        return Symmetry::ReflectRotate180;
    case Symmetry::ReflectRotate270:
        return Symmetry::ReflectRotate270;
    }
    throw std::invalid_argument("Unknown Go symmetry");
}

int32 transformAction(int32 action, int32 boardSize, Symmetry symmetry) {
    const int32 pass = checkedPointCount(boardSize);
    if (action == pass) {
        return pass;
    }
    if (action < 0 || action > pass) {
        throw std::invalid_argument("Go action out of range");
    }
    const auto [row, column] =
        transformCoordinate(action / boardSize, action % boardSize, boardSize, symmetry);
    return row * boardSize + column;
}

GoEncoding transformEncoding(const GoEncoding &encoding, Symmetry symmetry) {
    const GoEncodingShape shape = checkedEncodingShape(encoding.planes, encoding.boardSize);
    if (encoding.values.size() != shape.totalSize) {
        throw std::invalid_argument("Go encoding shape is inconsistent");
    }
    GoEncoding transformed{
        .planes = encoding.planes,
        .boardSize = encoding.boardSize,
        .values = std::vector<int8>(encoding.values.size(), 0),
    };
    for (int32 plane = 0; plane < encoding.planes; ++plane) {
        for (int32 row = 0; row < encoding.boardSize; ++row) {
            for (int32 column = 0; column < encoding.boardSize; ++column) {
                const auto [targetRow, targetColumn] =
                    transformCoordinate(row, column, encoding.boardSize, symmetry);
                const std::size_t target = shape.index(plane, targetRow, targetColumn);
                const std::size_t source = shape.index(plane, row, column);
                transformed.values[target] = encoding.values[source];
            }
        }
    }
    return transformed;
}

Symmetry GoSymmetryOperations::inverse(Symmetry symmetry) { return inverseSymmetry(symmetry); }

int32 GoSymmetryOperations::transformAction(int32 action, int32 boardSize, Symmetry symmetry) {
    return go::transformAction(action, boardSize, symmetry);
}

GoEncoding GoSymmetryOperations::transformEncoding(const GoEncoding &encoding, Symmetry symmetry) {
    return go::transformEncoding(encoding, symmetry);
}

} // namespace az::games::go
