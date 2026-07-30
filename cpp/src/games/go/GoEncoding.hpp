#pragma once

#include "common.hpp"
#include "games/go/GoState.hpp"

#include <cstdint>
#include <vector>

namespace az::games::go {

struct GoEncodingShape {
    int32 planes;
    int32 boardSize;
    std::size_t planeSize;
    std::size_t totalSize;

    [[nodiscard]] std::size_t index(int32 plane, int32 row, int32 column) const;
};

[[nodiscard]] GoEncodingShape checkedEncodingShape(int32 planes, int32 boardSize);

struct GoEncoding {
    int32 planes;
    int32 boardSize;
    std::vector<int8> values;

    [[nodiscard]] int8 at(int32 plane, int32 row, int32 column) const;
    [[nodiscard]] bool operator==(const GoEncoding &) const = default;
};

[[nodiscard]] GoEncoding canonicalEncoding(const GoState &state);

} // namespace az::games::go
