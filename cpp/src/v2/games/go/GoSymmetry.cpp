#include "games/go/GoSymmetry.hpp"

#include <stdexcept>

namespace az::v2::games::go {
namespace {

std::int32_t checked_point_count(std::int32_t board_size) {
    if (board_size != 7 && board_size != 9) {
        throw std::invalid_argument("Go board size must be 7 or 9");
    }
    return board_size * board_size;
}

} // namespace

Coordinate transform_coordinate(std::int32_t row, std::int32_t column, std::int32_t board_size,
                                Symmetry symmetry) {
    checked_point_count(board_size);
    if (row < 0 || row >= board_size || column < 0 || column >= board_size) {
        throw std::invalid_argument("Go coordinate out of range");
    }
    const std::int32_t maximum = board_size - 1;
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

Symmetry inverse_symmetry(Symmetry symmetry) {
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

std::int32_t transform_action(std::int32_t action, std::int32_t board_size, Symmetry symmetry) {
    const std::int32_t pass = checked_point_count(board_size);
    if (action == pass) {
        return pass;
    }
    if (action < 0 || action > pass) {
        throw std::invalid_argument("Go action out of range");
    }
    const auto [row, column] =
        transform_coordinate(action / board_size, action % board_size, board_size, symmetry);
    return row * board_size + column;
}

GoEncoding transform_encoding(const GoEncoding &encoding, Symmetry symmetry) {
    const std::int32_t plane_size = checked_point_count(encoding.board_size);
    if (encoding.planes < 1 || encoding.values.size() != static_cast<std::size_t>(encoding.planes) *
                                                             static_cast<std::size_t>(plane_size)) {
        throw std::invalid_argument("Go encoding shape is inconsistent");
    }
    GoEncoding transformed{
        .planes = encoding.planes,
        .board_size = encoding.board_size,
        .values = std::vector<std::int8_t>(encoding.values.size(), 0),
    };
    for (std::int32_t plane = 0; plane < encoding.planes; ++plane) {
        for (std::int32_t row = 0; row < encoding.board_size; ++row) {
            for (std::int32_t column = 0; column < encoding.board_size; ++column) {
                const auto [target_row, target_column] =
                    transform_coordinate(row, column, encoding.board_size, symmetry);
                const auto target = static_cast<std::size_t>(
                    plane * plane_size + target_row * encoding.board_size + target_column);
                transformed.values[target] = encoding.at(plane, row, column);
            }
        }
    }
    return transformed;
}

Symmetry GoSymmetryOperations::inverse(Symmetry symmetry) { return inverse_symmetry(symmetry); }

std::int32_t GoSymmetryOperations::transform_action(std::int32_t action, std::int32_t board_size,
                                                    Symmetry symmetry) {
    return go::transform_action(action, board_size, symmetry);
}

GoEncoding GoSymmetryOperations::transform_encoding(const GoEncoding &encoding, Symmetry symmetry) {
    return go::transform_encoding(encoding, symmetry);
}

} // namespace az::v2::games::go
