#pragma once

#include "common.hpp"
#include "games/go/GoEncoding.hpp"

#include <cstdint>

namespace az::games::go {

enum class Symmetry : int8 {
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
    int32 row;
    int32 column;

    [[nodiscard]] bool operator==(const Coordinate &) const = default;
};

[[nodiscard]] Coordinate transformCoordinate(int32 row, int32 column, int32 boardSize,
                                             Symmetry symmetry);
[[nodiscard]] Symmetry inverseSymmetry(Symmetry symmetry);
[[nodiscard]] int32 transformAction(int32 action, int32 boardSize, Symmetry symmetry);
[[nodiscard]] GoEncoding transformEncoding(const GoEncoding &encoding, Symmetry symmetry);

struct GoSymmetryOperations {
    using action_type = int32;
    using encoding_type = GoEncoding;
    using symmetry_type = Symmetry;

    [[nodiscard]] static Symmetry inverse(Symmetry symmetry);
    [[nodiscard]] static int32 transformAction(int32 action, int32 boardSize, Symmetry symmetry);
    [[nodiscard]] static GoEncoding transformEncoding(const GoEncoding &encoding,
                                                      Symmetry symmetry);
};

} // namespace az::games::go
