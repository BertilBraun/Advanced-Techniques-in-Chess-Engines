#include "position.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "BoardEncoding.hpp"
#include "MCTS/SearchTree.hpp"

namespace {
void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(const float actual, const float expected, const std::string &message) {
    if (std::abs(actual - expected) > 0.0001F) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", found " +
                                 std::to_string(actual));
    }
}

void initializeStockfish() {
    Bitboards::init();
    Position::init();
}

std::vector<MoveScore> legalMoveScores(const Board &board, const size_t count) {
    const std::vector<Move> &legalMoves = board.validMoves();
    require(legalMoves.size() >= count, "test position has too few legal moves");
    std::vector<MoveScore> scores;
    scores.reserve(count);
    const float policy = 1.0F / static_cast<float>(count);
    for (size_t index = 0; index < count; ++index) {
        scores.emplace_back(legalMoves[index], policy);
    }
    return scores;
}

void testExpansionCreatesOnlyChildRecords() {
    SearchTree tree(Board{}, 16);
    const NodeIndex rootIndex = tree.rootIndex();
    const std::vector<MoveScore> moves = legalMoveScores(tree.node(rootIndex).board, 5);
    tree.expand(rootIndex, moves);

    require(tree.liveNodeCount() == 1, "root expansion materialized child nodes");
    require(tree.totalChildCount() == moves.size(), "root expansion created wrong child count");
    for (const Child &child : tree.node(rootIndex).children) {
        require(child.node_index == INVALID_NODE_INDEX,
                "root expansion assigned a materialized node index");
    }
}

void testMaterializationConsumesOneSlot() {
    SearchTree tree(Board{}, 4);
    const NodeIndex rootIndex = tree.rootIndex();
    tree.expand(rootIndex, legalMoveScores(tree.node(rootIndex).board, 2));

    const NodeIndex childIndex = tree.materializeChild(rootIndex, 0);
    require(tree.liveNodeCount() == 2, "one edge did not consume exactly one node slot");
    require(tree.child(rootIndex, 0).node_index == childIndex,
            "materialized edge did not retain its node index");
    require(tree.child(rootIndex, 1).node_index == INVALID_NODE_INDEX,
            "materializing one edge changed another edge");
}

void testSelectionVirtualLossAndBackupStatistics() {
    SearchTree tree(Board{}, 4);
    const NodeIndex rootIndex = tree.rootIndex();
    tree.expand(rootIndex, legalMoveScores(tree.node(rootIndex).board, 2));

    const uint32 selectedChild = tree.bestChildIndex(rootIndex, 1.5F);
    const NodeIndex leafIndex = tree.materializeChild(rootIndex, selectedChild);
    tree.addVirtualLoss(leafIndex);

    require(tree.rootStatistics().number_of_visits == 1, "virtual loss missed root visit");
    requireNear(tree.rootStatistics().virtual_loss, 1.0F, "virtual loss missed root");
    require(tree.child(rootIndex, selectedChild).number_of_visits == 1,
            "virtual loss missed selected edge visit");
    requireNear(tree.child(rootIndex, selectedChild).virtual_loss, 1.0F,
                "virtual loss missed selected edge");

    tree.backPropagateAndRemoveVirtualLoss(leafIndex, 0.5F);
    require(tree.rootStatistics().number_of_visits == 1, "backup double-counted root visit");
    requireNear(tree.rootStatistics().virtual_loss, 0.0F, "backup left root virtual loss");
    requireNear(tree.rootStatistics().result_sum, -0.495F, "backup root turn discount changed");
    require(tree.child(rootIndex, selectedChild).number_of_visits == 1,
            "backup double-counted selected edge visit");
    requireNear(tree.child(rootIndex, selectedChild).virtual_loss, 0.0F,
                "backup left selected edge virtual loss");
    requireNear(tree.child(rootIndex, selectedChild).result_sum, 0.5F,
                "backup missed selected edge value");
}

void testRerootReclaimsAndReusesSlotsWithStaleDetection() {
    SearchTree tree(Board{}, 8);
    const NodeIndex oldRootIndex = tree.rootIndex();
    tree.expand(oldRootIndex, legalMoveScores(tree.node(oldRootIndex).board, 3));
    const NodeIndex retainedIndex = tree.materializeChild(oldRootIndex, 0);
    const NodeIndex discardedIndex = tree.materializeChild(oldRootIndex, 1);
    tree.expand(retainedIndex, legalMoveScores(tree.node(retainedIndex).board, 2));
    const NodeIndex retainedGrandchild = tree.materializeChild(retainedIndex, 0);
    require(tree.liveNodeCount() == 4, "reroot fixture has unexpected live-node count");

    const NodeIndex newRootIndex = tree.reroot(0);
    require(newRootIndex == retainedIndex, "reroot changed the retained node index");
    require(tree.liveNodeCount() == 2, "reroot did not reclaim old and discarded nodes");
    require(tree.node(newRootIndex).parent_index == INVALID_NODE_INDEX,
            "reroot retained the former parent index");
    require(tree.child(newRootIndex, 0).node_index == retainedGrandchild,
            "reroot discarded the selected subtree");

    bool oldRootWasStale = false;
    try {
        static_cast<void>(tree.node(oldRootIndex));
    } catch (const std::logic_error &) {
        oldRootWasStale = true;
    }
    require(oldRootWasStale, "reclaimed old root index was not detected as stale");

    bool discardedWasStale = false;
    try {
        static_cast<void>(tree.node(discardedIndex));
    } catch (const std::logic_error &) {
        discardedWasStale = true;
    }
    require(discardedWasStale, "reclaimed branch index was not detected as stale");

    const NodeIndex reusedIndex = tree.materializeChild(newRootIndex, 1);
    require(static_cast<uint32>(reusedIndex) == static_cast<uint32>(oldRootIndex),
            "free-list did not reuse the most recently reclaimed slot");
    require(reusedIndex != oldRootIndex, "reused slot did not advance its generation");
}

void testDiscountPrunesToTightCapacityInvariant() {
    constexpr uint32 searchBudget = 8;
    constexpr uint32 parallelSearches = 2;
    constexpr uint32 capacity = searchBudget + parallelSearches + 1;
    SearchTree tree(Board{}, capacity);

    for (int turn = 0; turn < 6; ++turn) {
        if (tree.rootStatistics().number_of_visits > 0) {
            tree.discount(0.25F);
            require(tree.liveNodeCount() <= tree.rootStatistics().number_of_visits + 1,
                    "discount left too many materialized nodes for the remaining budget");
        }

        while (tree.rootStatistics().number_of_visits < searchBudget) {
            NodeIndex selectedIndex = tree.rootIndex();
            while (tree.node(selectedIndex).isExpanded()) {
                selectedIndex = tree.materializeChild(selectedIndex, 0);
            }
            if (tree.node(selectedIndex).isTerminal()) {
                tree.backPropagate(selectedIndex,
                                   getBoardResultScore(tree.node(selectedIndex).board));
                continue;
            }
            tree.addVirtualLoss(selectedIndex);
            tree.backPropagateAndRemoveVirtualLoss(selectedIndex, 0.0F);
            const std::vector<Move> &moves = tree.node(selectedIndex).board.validMoves();
            if (!moves.empty()) {
                tree.expand(selectedIndex, {{moves.front(), 1.0F}});
            }
            require(tree.liveNodeCount() <= capacity, "search exceeded its tight arena capacity");
        }

        require(!tree.node(tree.rootIndex()).children.empty(),
                "repeated-capacity fixture reached an unexpected terminal root");
        static_cast<void>(tree.reroot(0));
        require(tree.liveNodeCount() <= capacity, "reroot exceeded its tight arena capacity");
    }
}

void testTerminalAndUnmaterializedReroot() {
    SearchTree terminalTree(Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), 2);
    require(terminalTree.node(terminalTree.rootIndex()).isTerminal(),
            "terminal fixture was not terminal");
    terminalTree.backPropagate(
        terminalTree.rootIndex(),
        getBoardResultScore(terminalTree.node(terminalTree.rootIndex()).board));
    require(terminalTree.rootStatistics().number_of_visits == 1,
            "terminal root backup missed its root visit");

    SearchTree tree(Board{}, 3);
    const NodeIndex oldRootIndex = tree.rootIndex();
    const std::vector<MoveScore> moves = legalMoveScores(tree.node(oldRootIndex).board, 2);
    Board expectedBoard = tree.node(oldRootIndex).board;
    expectedBoard.makeMove(moves[1].first);
    tree.expand(oldRootIndex, moves);
    require(tree.child(oldRootIndex, 1).node_index == INVALID_NODE_INDEX,
            "unvisited reroot edge was unexpectedly materialized");

    static_cast<void>(tree.reroot(1));
    require(tree.liveNodeCount() == 1, "unmaterialized reroot did not reclaim the old root");
    require(tree.node(tree.rootIndex()).board.fen() == expectedBoard.fen(),
            "unmaterialized reroot constructed the wrong Board");
    require(tree.rootStatistics().number_of_visits == 0, "unvisited reroot edge gained a visit");
}
} // namespace

int main() {
    initializeStockfish();
    testExpansionCreatesOnlyChildRecords();
    testMaterializationConsumesOneSlot();
    testSelectionVirtualLossAndBackupStatistics();
    testRerootReclaimsAndReusesSlotsWithStaleDetection();
    testDiscountPrunesToTightCapacityInvariant();
    testTerminalAndUnmaterializedReroot();
    std::cout << "SearchTree arena tests passed\n";
    return 0;
}
