#pragma once

#include "common.hpp"

#include "AlphaMCTSNode.hpp"
#include "BoardEncoding.hpp"

struct SelfPlayGameMemory {
    Board board;
    torch::Tensor actionProbabilities;
    Color turn;

    SelfPlayGameMemory(const Board &board, const torch::Tensor &actionProbabilities, Color turn)
        : board(board), actionProbabilities(actionProbabilities), turn(turn) {}
};

class SelfPlayGame {
public:
    SelfPlayGame() : board(), root(nullptr), node(nullptr) {}

    AlphaMCTSNode *getBestChildOrBackPropagate(float cParam) {
        AlphaMCTSNode *currentNode = root;

        while (currentNode && currentNode->isFullyExpanded()) {
            currentNode = &currentNode->bestChild(cParam);
        }

        if (currentNode && currentNode->isTerminalNode()) {
            currentNode->backPropagate(getBoardResultScore(currentNode->board));
            return nullptr;
        }

        return currentNode;
    }

    Board board;
    std::vector<SelfPlayGameMemory> memory;
    AlphaMCTSNode *root;
    AlphaMCTSNode *node;
};