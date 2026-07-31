#pragma once

#include "common.hpp"

#include <optional>
#include <position.h>

namespace az::games::chess {

inline constexpr int32 CHESS_ACTION_COUNT = 1880;

[[nodiscard]] int32 encodeMove(Stockfish::Move move, Stockfish::Color player);
[[nodiscard]] std::optional<Stockfish::Move> decodeLegalMove(int32 action,
                                                             const Stockfish::Position &position);

} // namespace az::games::chess
