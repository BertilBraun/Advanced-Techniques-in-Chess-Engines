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

    const std::vector<Move> &legalMoves = root->board().validMoves();
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
    require(!first->front()->hasMaterializedBoard(),
            "expansion unnecessarily materialized a child board");
    first->front()->materializeBoard();
    require(first->front()->hasMaterializedBoard(), "selected child board was not materialized");
}

void testSnapshotSurvivesRerooting() {
    const std::shared_ptr<EvalMCTSNode> root = EvalMCTSNode::createRoot(Board{});
    const std::vector<Move> &legalMoves = root->board().validMoves();
    root->expand({
        {legalMoves[0], 0.5F},
        {legalMoves[1], 0.5F},
    });

    const EvalMCTSNode::ChildSnapshot snapshot = root->children();
    require(snapshot != nullptr, "expanded root did not publish children");

    const std::shared_ptr<EvalMCTSNode> selectedChild = snapshot->front();
    selectedChild->materializeBoard();
    const std::vector<Move> &replyMoves = selectedChild->board().validMoves();
    selectedChild->expand({{replyMoves.front(), 1.0F}});
    const EvalMCTSNode::ChildSnapshot replies = selectedChild->children();
    require(replies != nullptr, "selected child did not retain a reply subtree");

    const std::shared_ptr<EvalMCTSNode> newRoot = root->makeNewRoot(0);
    require(root->children() == nullptr, "old root retained its published children");
    require(snapshot->size() == 2, "existing snapshot was invalidated by rerooting");
    require(snapshot->front() == newRoot, "rerooting selected an unexpected child");
    require(newRoot->parent.expired(), "new root retained its former parent");
    require(newRoot->children() == replies, "rerooting discarded the reusable subtree");
}

void testBackupAlternatesPerspectiveWithoutDiscount() {
    const std::shared_ptr<EvalMCTSNode> root = EvalMCTSNode::createRoot(Board{});
    const Move move = root->board().validMoves().front();
    const WdlPrediction outcome{0.6F, 0.3F, 0.1F};
    root->expand({{move, 1.0F}}, outcome);

    const EvalMCTSNode::ChildSnapshot children = root->children();
    require(children != nullptr, "expanded root had no children");
    const std::shared_ptr<EvalMCTSNode> child = children->front();
    child->addVirtualLoss();
    child->backPropagateAndRemoveVirtualLoss(0.5F);

    require(child->result_sum.load() == 0.5F, "child value was not child-to-move perspective");
    require(root->result_sum.load() == -0.5F, "root value did not alternate perspective");
    require(root->networkOutcome.has_value(), "root WDL prediction was not retained");
    require(root->networkOutcome->draw == outcome.draw, "root WDL prediction changed");
}
} // namespace

int main() {
    initializeStockfish();
    testChildrenUseStableImmutableSnapshot();
    testSnapshotSurvivesRerooting();
    testBackupAlternatesPerspectiveWithoutDiscount();
    std::cout << "Eval MCTS child snapshot tests passed\n";
    return 0;
}
