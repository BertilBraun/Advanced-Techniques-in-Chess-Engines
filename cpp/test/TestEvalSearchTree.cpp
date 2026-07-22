#include "position.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "BoardEncoding.hpp"
#include "GameHistory.hpp"
#include "MCTS/EvalSearchTree.hpp"

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

std::vector<MoveScore> firstMoves(const Board &board, const std::size_t count) {
    const std::vector<Move> &legalMoves = board.validMoves();
    std::vector<MoveScore> moves;
    for (std::size_t index = 0; index < std::min(count, legalMoves.size()); ++index) {
        moves.emplace_back(legalMoves[index], 1.0F / static_cast<float>(count));
    }
    return moves;
}

void testLazyExpansionGrowthAndWdl(const EvalEdgeStorage storage) {
    EvalSearchTree tree(Board{}, 1, storage);
    const WdlPrediction outcome{0.6F, 0.3F, 0.1F};
    tree.expand(tree.rootIndex(), firstMoves(tree.rootBoard(), 3), outcome);
    require(tree.liveNodeCount() == 1, "expansion materialized child boards");
    require(tree.totalEdgeCount() == 3, "expansion created the wrong edge count");
    require(tree.node(tree.rootIndex()).network_outcome.has_value(), "expansion discarded WDL");

    const EvalNodeIndex child = tree.materializeChild(tree.rootIndex(), 0);
    require(tree.liveNodeCount() == 2, "child materialization did not allocate one node");
    require(tree.capacity() >= 2, "node arena did not grow");
    require(tree.node(child).board.fen() != tree.rootBoard().fen(),
            "materialized child did not apply its move");
}

void testPerspectiveBackupAndCancellation() {
    EvalSearchTree tree(Board{});
    tree.expand(tree.rootIndex(), firstMoves(tree.rootBoard(), 1));
    const EvalNodeIndex child = tree.materializeChild(tree.rootIndex(), 0);

    tree.addVirtualLoss(child);
    require(tree.rootStatistics().visits == 1, "virtual loss did not reserve root visit");
    tree.backPropagateAndRemoveVirtualLoss(child, 0.5F);
    require(tree.edge(tree.rootIndex(), 0).statistics.result_sum == 0.5F,
            "child result used the wrong perspective");
    require(tree.rootStatistics().result_sum == -0.5F, "root result did not alternate perspective");
    require(tree.rootStatistics().virtual_loss == 0, "backup retained virtual loss");

    tree.addVirtualLoss(child);
    tree.cancelVirtualLoss(child);
    require(tree.rootStatistics().visits == 1, "cancel did not restore visit count");
    require(tree.rootStatistics().virtual_loss == 0, "cancel retained virtual loss");
}

void testFinalPreferenceUsesVisitsWithDeterministicTieBreak() {
    EvalSearchTree tree(Board{});
    tree.expand(tree.rootIndex(), firstMoves(tree.rootBoard(), 3));
    tree.edge(tree.rootIndex(), 0).statistics.visits = 10;
    tree.edge(tree.rootIndex(), 1).statistics.visits = 20;
    tree.edge(tree.rootIndex(), 2).statistics.visits = 15;
    require(tree.preferredRootEdge() == 1, "final preference did not maximize visits");

    tree.edge(tree.rootIndex(), 2).statistics.visits = 20;
    tree.edge(tree.rootIndex(), 1).policy = 0.2F;
    tree.edge(tree.rootIndex(), 2).policy = 0.8F;
    require(tree.preferredRootEdge() == 2, "visit tie did not prefer the larger policy");

    tree.edge(tree.rootIndex(), 1).policy = 0.8F;
    const std::uint32_t expected = toString(tree.edge(tree.rootIndex(), 1).move) <
                                           toString(tree.edge(tree.rootIndex(), 2).move)
                                       ? 1U
                                       : 2U;
    require(tree.preferredRootEdge() == expected, "policy tie did not use UCI lexical order");
}

void testConfiguredCapacityLimit() {
    EvalSearchTree tree(Board{}, 1, EvalEdgeStorage::UnsynchronizedPool, 2);
    tree.expand(tree.rootIndex(), firstMoves(tree.rootBoard(), 2));
    static_cast<void>(tree.materializeChild(tree.rootIndex(), 0));
    bool rejectedGrowth = false;
    try {
        static_cast<void>(tree.materializeChild(tree.rootIndex(), 1));
    } catch (const std::overflow_error &) {
        rejectedGrowth = true;
    }
    require(rejectedGrowth, "configured arena limit did not reject growth");
    require(tree.capacity() == tree.maximumCapacity(), "arena exceeded its configured limit");
}

void testRerootRetainsLightlySearchedReplyAndRejectsStaleIndices() {
    EvalSearchTree tree(Board{}, 2);
    tree.expand(tree.rootIndex(), firstMoves(tree.rootBoard(), 2));
    const EvalNodeIndex oldRoot = tree.rootIndex();
    const EvalNodeIndex popular = tree.materializeChild(oldRoot, 0);
    const EvalNodeIndex quiet = tree.materializeChild(oldRoot, 1);
    for (int search = 0; search < 8; ++search) {
        tree.backPropagate(popular, 0.25F);
    }
    for (int search = 0; search < 2; ++search) {
        tree.backPropagate(quiet, -0.25F);
    }

    tree.expand(quiet, firstMoves(tree.node(quiet).board, 2), WdlPrediction{0.2F, 0.7F, 0.1F});
    const EvalNodeIndex reply = tree.materializeChild(quiet, 0);
    tree.backPropagate(reply, 0.5F);
    const std::uint32_t retainedVisits = tree.edge(oldRoot, 1).statistics.visits;
    static_cast<void>(tree.reroot(1));

    require(tree.rootIndex() == quiet, "reroot selected the wrong child");
    require(tree.rootStatistics().visits == retainedVisits,
            "reroot lost visits on the lightly searched human reply");
    require(tree.node(tree.rootIndex()).network_outcome.has_value(),
            "reroot lost the retained subtree WDL");
    require(tree.edge(tree.rootIndex(), 0).child == reply,
            "reroot discarded the explored descendant");
    require(tree.liveNodeCount() == 2, "reroot did not reclaim discarded materialized nodes");

    bool rejectedStaleRoot = false;
    try {
        static_cast<void>(tree.node(oldRoot));
    } catch (const std::logic_error &) {
        rejectedStaleRoot = true;
    }
    require(rejectedStaleRoot, "reroot left the old root index usable");
}

void testTerminalAndHistoryAwareBoards() {
    const std::string startingFen = Board{}.fen();
    const Board checkmate = replayMoves(startingFen, {"f2f3", "e7e5", "g2g4", "d8h4"});
    EvalSearchTree terminalTree(checkmate);
    require(terminalTree.node(terminalTree.selectLeaf(1.0F)).isTerminal(),
            "terminal root was not recognized");
    terminalTree.backPropagate(terminalTree.rootIndex(),
                               getBoardResultScore(terminalTree.rootBoard()));
    require(terminalTree.rootStatistics().visits == 1,
            "terminal result did not count as a completed search");

    const Board repeated =
        replayMoves(startingFen, {"g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"});
    EvalSearchTree repetitionTree(repeated);
    require(repetitionTree.rootBoard().isGameOver(),
            "arena root lost repetition history from the replayed Board");
}

void testRepeatedRerootReusesReleasedSlots(const EvalEdgeStorage storage) {
    EvalSearchTree tree(Board{}, 2, storage);
    for (int ply = 0; ply < 40 && !tree.rootBoard().isGameOver(); ++ply) {
        const std::vector<MoveScore> moves = firstMoves(tree.rootBoard(), 2);
        tree.expand(tree.rootIndex(), moves);
        const EvalNodeIndex oldRoot = tree.rootIndex();
        static_cast<void>(tree.materializeChild(oldRoot, 0));
        EvalNodeIndex discarded = INVALID_EVAL_NODE_INDEX;
        if (moves.size() > 1) {
            discarded = tree.materializeChild(oldRoot, 1);
        }
        static_cast<void>(tree.reroot(0));
        require(tree.liveNodeCount() == 1, "reroot failed to reclaim the discarded branch");
        if (discarded != INVALID_EVAL_NODE_INDEX) {
            bool rejectedDiscarded = false;
            try {
                static_cast<void>(tree.node(discarded));
            } catch (const std::logic_error &) {
                rejectedDiscarded = true;
            }
            require(rejectedDiscarded, "discarded subtree index remained live");
        }
    }
    require(tree.maximumLiveNodeCount() <= 3, "serial reroot unexpectedly accumulated live nodes");
    require(tree.capacity() <= 4, "released slots were not reused across a long game");
}
} // namespace

int main() {
    initializeStockfish();
    testLazyExpansionGrowthAndWdl(EvalEdgeStorage::NewDelete);
    testLazyExpansionGrowthAndWdl(EvalEdgeStorage::UnsynchronizedPool);
    testPerspectiveBackupAndCancellation();
    testFinalPreferenceUsesVisitsWithDeterministicTieBreak();
    testConfiguredCapacityLimit();
    testRerootRetainsLightlySearchedReplyAndRejectsStaleIndices();
    testTerminalAndHistoryAwareBoards();
    testRepeatedRerootReusesReleasedSlots(EvalEdgeStorage::NewDelete);
    testRepeatedRerootReusesReleasedSlots(EvalEdgeStorage::UnsynchronizedPool);
    std::cout << "Evaluation search-tree tests passed\n";
    return 0;
}
