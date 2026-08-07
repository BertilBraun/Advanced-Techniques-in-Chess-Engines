#pragma once

#include "games/chess/ChessGameContract.hpp"
#include "search/SearchTree.hpp"

#include <cstdint>
#include <string>
#include <vector>

using ChessSearchRoot = GameSearchRoot<ChessGameContract>;

struct ChessSearchChild {
    std::string move;
    int encoded_move;
    float raw_policy;
    float policy;
    std::uint32_t visits;
    float result_sum;
    float virtual_loss;
    bool is_materialized;
};

[[nodiscard]] ChessSearchRoot createChessSearchRoot(Board board, std::uint32_t arenaCapacity);
[[nodiscard]] std::vector<ChessSearchChild> chessSearchChildren(const ChessSearchRoot &root);
[[nodiscard]] ChessSearchRoot rerootChessSearch(ChessSearchRoot root, std::uint32_t childIndex);
[[nodiscard]] std::string describeChessSearchRoot(const ChessSearchRoot &root);
