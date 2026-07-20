#include "position.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "MCTS/EvalMCTSNode.h"

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void initializeStockfish() {
    Bitboards::init();
    Position::init();
}

void testChildrenUseStableImmutableSnapshot() {
    const std::shared_ptr<EvalMCTSNode> root = EvalMCTSNode::createRoot(Board{});
    require(root->children() == nullptr, "new root unexpectedly has a child snapshot");

    const std::vector<Move> &legalMoves = root->board.validMoves();
    const std::vector<MoveScore> moves = {
        {legalMoves[0], 0.5F},
        {legalMoves[1], 0.5F},
    };
    root->expand(moves);

    const EvalMCTSNode::ChildSnapshot first = root->children();
    const EvalMCTSNode::ChildSnapshot second = root->children();
    require(first != nullptr, "expanded root did not publish children");
    require(first == second, "successive reads did not share the published child vector");
    require(first->size() == moves.size(), "published child vector has the wrong size");
}

void testSnapshotSurvivesRerooting() {
    const std::shared_ptr<EvalMCTSNode> root = EvalMCTSNode::createRoot(Board{});
    const std::vector<Move> &legalMoves = root->board.validMoves();
    root->expand({
        {legalMoves[0], 0.5F},
        {legalMoves[1], 0.5F},
    });

    const EvalMCTSNode::ChildSnapshot snapshot = root->children();
    require(snapshot != nullptr, "expanded root did not publish children");

    const std::shared_ptr<EvalMCTSNode> newRoot = root->makeNewRoot(0);
    require(root->children() == nullptr, "old root retained its published children");
    require(snapshot->size() == 2, "existing snapshot was invalidated by rerooting");
    require(snapshot->front() == newRoot, "rerooting selected an unexpected child");
    require(newRoot->parent.expired(), "new root retained its former parent");
}
} // namespace

int main() {
    initializeStockfish();
    testChildrenUseStableImmutableSnapshot();
    testSnapshotSurvivesRerooting();
    std::cout << "Eval MCTS child snapshot tests passed\n";
    return 0;
}
