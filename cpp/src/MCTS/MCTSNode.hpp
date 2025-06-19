#pragma once

#include "common.hpp"

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    static std::shared_ptr<MCTSNode> createRoot(const std::string &fen);

    [[nodiscard]] bool isTerminal() const { return board.isGameOver(); }

    [[nodiscard]] bool isExpanded() const { return !children.empty(); }

    [[nodiscard]] float ucb(float uCommon, float parentQ) const;

    void expand(const std::vector<MoveScore> &moves_with_scores);

    void backPropagate(float result);

    void backPropagateAndRemoveVirtualLoss(float result);

    void addVirtualLoss();

    [[nodiscard]] std::shared_ptr<MCTSNode> bestChild(float cParam) const;

    [[nodiscard]] bool operator==(const MCTSNode &other) const;

    /* prune old tree, return chosen child as new root */
    [[nodiscard]] std::shared_ptr<MCTSNode> makeNewRoot(std::size_t childIdx, float discount);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] int maxDepth() const;

    std::weak_ptr<MCTSNode> parent;
    std::vector<std::shared_ptr<MCTSNode>> children;

    Board board;
    Move move_to_get_here = Move::null();
    int number_of_visits = 0;
    float virtual_loss = 0.0;
    float result_sum = 0.0;
    float policy = 0.0;

private:
    MCTSNode(const std::string &fen, float policy, Move move, std::weak_ptr<MCTSNode> parent);
};
