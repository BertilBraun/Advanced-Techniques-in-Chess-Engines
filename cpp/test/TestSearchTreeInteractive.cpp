#include "position.h"

#include "MCTS/SearchTree.hpp"
#include "games/chess/ChessHistory.hpp"

#include <iostream>

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<MoveScore> firstMoves(const Board &board, const std::size_t count) {
    const std::vector<Move> &legalMoves = board.validMoves();
    std::vector<MoveScore> moves;
    for (std::size_t index = 0; index < std::min(count, legalMoves.size()); ++index) {
        moves.emplace_back(legalMoves[index], 1.0F / static_cast<float>(count));
    }
    return moves;
}

void testGrowthReservationsAndWdl() {
    SearchTree tree(Board{}, 1, 4, 1.0F);
    tree.expand(tree.rootIndex(), firstMoves(tree.node(tree.rootIndex()).board, 2),
                WdlPrediction{0.6F, 0.3F, 0.1F});
    require(tree.node(tree.rootIndex()).network_outcome.has_value(), "expansion discarded WDL");
    const NodeIndex child = tree.materializeChild(tree.rootIndex(), 0);
    require(tree.capacity() == 2, "interactive arena did not grow");
    tree.reserveLeaf(child);
    require(tree.evaluatingNodeCount() == 1 && tree.totalVirtualLoss() == 2,
            "leaf reservation did not reach the root");
    tree.completeReservation(child, 0.5F);
    require(tree.evaluatingNodeCount() == 0 && tree.totalVirtualLoss() == 0,
            "completed reservation retained transient state");
    require(tree.rootStatistics().result_sum == -0.5F,
            "interactive backup used the wrong perspective");
}

void testSelectionAvoidsReservedLeaves() {
    SearchTree tree(Board{}, 2, 16, 1.0F);
    tree.expand(tree.rootIndex(), firstMoves(tree.node(tree.rootIndex()).board, 2));
    const NodeIndex first = tree.selectAvailableLeaf(1.5F);
    tree.reserveLeaf(first);
    const NodeIndex second = tree.selectAvailableLeaf(1.5F);
    require(second != INVALID_NODE_INDEX && second != first,
            "parallel selection reused a reserved leaf");
    tree.cancelReservation(first);
}

void testRerootRetainsSubtreeAndRejectsStaleIndices() {
    SearchTree tree(Board{}, 2, 16, 1.0F);
    const NodeIndex oldRoot = tree.rootIndex();
    tree.expand(oldRoot, firstMoves(tree.node(oldRoot).board, 2));
    const NodeIndex retained = tree.materializeChild(oldRoot, 1);
    tree.expand(retained, firstMoves(tree.node(retained).board, 2),
                WdlPrediction{0.2F, 0.7F, 0.1F});
    const NodeIndex reply = tree.materializeChild(retained, 0);
    tree.backPropagate(reply, 0.5F);
    static_cast<void>(tree.reroot(1));
    require(tree.rootIndex() == retained && tree.node(retained).children[0].node_index == reply,
            "reroot discarded the retained interactive subtree");
    require(tree.node(retained).network_outcome.has_value(), "reroot discarded retained WDL");
    bool rejectedStaleRoot = false;
    try {
        static_cast<void>(tree.node(oldRoot));
    } catch (const std::logic_error &) {
        rejectedStaleRoot = true;
    }
    require(rejectedStaleRoot, "reroot left the old root index usable");
}

void testHistoryAwareTerminalRoot() {
    const std::string startingFen = Board{}.fen();
    const Board checkmate = replayMoves(startingFen, {"f2f3", "e7e5", "g2g4", "d8h4"});
    SearchTree tree(checkmate, 1, 4, 1.0F);
    const NodeIndex leaf = tree.selectAvailableLeaf(1.0F);
    require(tree.node(leaf).isTerminal(), "terminal interactive root was not recognized");
}
} // namespace

int main() {
    Bitboards::init();
    Position::init();
    testGrowthReservationsAndWdl();
    testSelectionAvoidsReservedLeaves();
    testRerootRetainsSubtreeAndRejectsStaleIndices();
    testHistoryAwareTerminalRoot();
    std::cout << "Interactive search-tree tests passed\n";
    return 0;
}
