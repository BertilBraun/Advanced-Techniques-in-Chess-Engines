#pragma once

#include "common.hpp"

#include "BoardEncoding.hpp"

std::vector<std::vector<std::vector<float>>> testEncode(std::string fen) {
    // Encode the board to tensor, then return it
    const Board board(fen);
    CompressedEncodedBoard encodedBoard = encodeBoard(&board);
    torch::Tensor tensor = toTensor(encodedBoard);
    std::vector<std::vector<std::vector<float>>> result;
    for (int i : range(tensor.size(0))) {
        std::vector<std::vector<float>> channel;
        for (int j : range(tensor.size(1))) {
            std::vector<float> row;
            for (int k : range(tensor.size(2))) {
                row.push_back(tensor[i][j][k].item<float>());
            }
            channel.push_back(row);
        }
        result.push_back(channel);
    }
    return result;
}
