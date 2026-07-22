#pragma once

#include "Board.h"

[[nodiscard]] Move findLegalMove(const Board &board, const std::string &moveUci);

[[nodiscard]] Board replayMoves(const std::string &startingFen,
                                const std::vector<std::string> &movesUci);
