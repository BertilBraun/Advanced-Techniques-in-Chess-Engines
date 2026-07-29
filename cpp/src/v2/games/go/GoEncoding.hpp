#pragma once

#include "games/go/GoState.hpp"

#include <cstdint>
#include <vector>

namespace az::v2::games::go {

struct GoEncoding {
    std::int32_t planes;
    std::int32_t board_size;
    std::vector<std::int8_t> values;

    [[nodiscard]] std::int8_t at(std::int32_t plane, std::int32_t row, std::int32_t column) const;
    [[nodiscard]] bool operator==(const GoEncoding &) const = default;
};

[[nodiscard]] GoEncoding canonical_encoding(const GoState &state);

} // namespace az::v2::games::go
