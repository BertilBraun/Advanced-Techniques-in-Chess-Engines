#pragma once

#include "games/go/GoEncoding.hpp"

#include <cstdint>

namespace az::v2::games::go {

enum class Symmetry : std::int8_t {
    Identity = 0,
    Rotate90 = 1,
    Rotate180 = 2,
    Rotate270 = 3,
    Reflect = 4,
    ReflectRotate90 = 5,
    ReflectRotate180 = 6,
    ReflectRotate270 = 7,
};

struct Coordinate {
    std::int32_t row;
    std::int32_t column;

    [[nodiscard]] bool operator==(const Coordinate &) const = default;
};

[[nodiscard]] Coordinate transform_coordinate(std::int32_t row, std::int32_t column,
                                              std::int32_t board_size, Symmetry symmetry);
[[nodiscard]] Symmetry inverse_symmetry(Symmetry symmetry);
[[nodiscard]] std::int32_t transform_action(std::int32_t action, std::int32_t board_size,
                                            Symmetry symmetry);
[[nodiscard]] GoEncoding transform_encoding(const GoEncoding &encoding, Symmetry symmetry);

struct GoSymmetryOperations {
    using action_type = std::int32_t;
    using encoding_type = GoEncoding;
    using symmetry_type = Symmetry;

    [[nodiscard]] static Symmetry inverse(Symmetry symmetry);
    [[nodiscard]] static std::int32_t transform_action(std::int32_t action, std::int32_t board_size,
                                                       Symmetry symmetry);
    [[nodiscard]] static GoEncoding transform_encoding(const GoEncoding &encoding,
                                                       Symmetry symmetry);
};

} // namespace az::v2::games::go
