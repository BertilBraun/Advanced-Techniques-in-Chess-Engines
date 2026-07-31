#pragma once

#include "common.hpp"

#include <array>

namespace az::games::chess {

class ChessState;

inline constexpr int32 CHESS_BOARD_SIZE = 8;
inline constexpr int32 CHESS_ENCODING_PLANES = 29;

struct ChessEncoding {
    std::array<int8, CHESS_ENCODING_PLANES * CHESS_BOARD_SIZE * CHESS_BOARD_SIZE> values;

    [[nodiscard]] int8 at(int32 plane, int32 row, int32 column) const;
    [[nodiscard]] bool operator==(const ChessEncoding &) const = default;
};

[[nodiscard]] ChessEncoding canonicalEncoding(const ChessState &state);

} // namespace az::games::chess
