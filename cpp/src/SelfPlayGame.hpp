#pragma once

#include "common.hpp"

#include "AlphaMCTSNode.hpp"
#include "BoardEncoding.hpp"

struct SelfPlayGameMemory {
    Board board;
    torch::Tensor actionProbabilities;

    SelfPlayGameMemory(const Board &board, const torch::Tensor &actionProbabilities)
        : board(board), actionProbabilities(actionProbabilities) {}
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
            auto result = getBoardResultScore(currentNode->board);
            assert(result.has_value());
            currentNode->backPropagate(result.value());
            return nullptr;
        }

        return currentNode;
    }

    void init(const std::vector<std::pair<Move, float>> &moves) {
        root = AlphaMCTSNode::root(board);
        root->expand(moves);
    }

    void push(Move move) {
        board.push(move);
        delete root;
    }

    Board board;
    std::vector<SelfPlayGameMemory> memory;
    AlphaMCTSNode *root;
    AlphaMCTSNode *node;
};