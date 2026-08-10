#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "games/go/GoGame.hpp"
#include "search/ForcedPlayouts.hpp"
#include "search/SearchTree.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <SearchGame Game> void exercise_tree(typename Game::State position) {
    GameSearchTree<Game> tree(std::move(position), 1, 16);
    const auto legalActions = Game::legalActions(tree.root().position);
    SearchInferenceResult<Game> inference{{}, {0.5F, 0.0F, 0.5F}};
    inference.actions.reserve(legalActions.size());
    for (const typename Game::Action action : legalActions) {
        inference.actions.emplace_back(action, 1.0F / static_cast<float>(legalActions.size()));
    }
    tree.expand(tree.rootIndex(), inference);
    require(tree.root().children.size() == Game::legalActions(tree.root().position).size(),
            "Shared game tree did not create every legal edge");
    const std::size_t leaf = *tree.selectAvailableLeaf(1.5F);
    require(leaf != tree.rootIndex(), "Shared game tree did not materialize a child");
    tree.backPropagate(leaf, 0.5F);
    require(tree.root().visits == 1, "Shared game tree did not backpropagate to the root");
    require(tree.liveNodeCount() == 2, "Shared game tree created an unexpected node count");
    require(tree.capacity() >= 2, "Shared game tree did not grow its reusable arena");
    const auto &leafNode = tree.node(leaf);
    const int selectedAction = Game::Encoding::actionId(
        tree.root().children[*leafNode.parent_edge_index].action, tree.root().position);
    tree.reroot(selectedAction);
    require(tree.root().visits == 1, "Shared game tree discarded retained child statistics");
    require(tree.liveNodeCount() == 1, "Shared game tree did not reclaim the previous root");
    tree.reset();
    require(tree.root().visits == 0, "Shared game tree reset retained old statistics");
}

std::size_t selectedRootEdgeAfterOneVisit(const float rootValue, const float fpuReduction,
                                          const float forcedPlayoutCoefficient = 0.0F) {
    GameSearchTree<Go7Game> tree(
        GoPosition<7>(GoRules{.komi_half_points = 15, .maximum_moves = 196}), 1, 64);
    const auto legalActions = Go7Game::legalActions(tree.root().position);
    SearchInferenceResult<Go7Game> inference{{}, {0.5F, 0.0F, 0.5F}};
    for (const GoAction action : legalActions) {
        inference.actions.emplace_back(action, 1.0F / static_cast<float>(legalActions.size()));
    }
    tree.expand(tree.rootIndex(), inference);
    const std::size_t firstLeaf = *tree.selectAvailableLeaf(0.1F);
    tree.backPropagate(firstLeaf, -rootValue);

    const std::size_t selectedLeaf =
        *tree.selectAvailableLeaf(0.1F, fpuReduction, forcedPlayoutCoefficient);
    return *tree.node(selectedLeaf).parent_edge_index;
}

void testForcedPlayoutMath() {
    require(
        requiresForcedRootVisit({.prior = 0.25F, .visits = 7, .child_mean_value = 0.0F}, 100, 2.0F),
        "Prior-scaled forced playout stopped below its threshold");
    require(!requiresForcedRootVisit({.prior = 0.25F, .visits = 8, .child_mean_value = 0.0F}, 100,
                                     2.0F),
            "Prior-scaled forced playout continued above its threshold");
    require(!requiresForcedRootVisit({.prior = 0.25F, .visits = 0, .child_mean_value = 0.0F}, 100,
                                     0.0F),
            "Disabled forced playout changed selection");

    const std::vector<RootChildSearchStatistics> unsupported = {
        {.prior = 0.99F, .visits = 90, .child_mean_value = 0.0F},
        {.prior = 0.01F, .visits = 10, .child_mean_value = 0.5F},
    };
    const std::vector<std::uint32_t> pruned = prunedRootPolicyVisits(unsupported, 100, 1.5F, true);
    require(pruned[0] == 90 && pruned[1] < 10,
            "Policy pruning did not remove unsupported forced excess");
    const std::vector<std::uint32_t> disabled =
        prunedRootPolicyVisits(unsupported, 100, 1.5F, false);
    require(disabled[0] == 90 && disabled[1] == 10,
            "Disabled policy pruning changed actual visits");

    const std::vector<RootChildSearchStatistics> supported = {
        {.prior = 0.6F, .visits = 60, .child_mean_value = 0.0F},
        {.prior = 0.4F, .visits = 40, .child_mean_value = 0.0F},
    };
    require(prunedRootPolicyVisits(supported, 100, 1.5F, true)[1] == 40,
            "Policy pruning removed visits supported by ordinary PUCT");

    const std::vector<RootChildSearchStatistics> winningAlternative = {
        {.prior = 0.99F, .visits = 90, .child_mean_value = 0.0F},
        {.prior = 0.01F, .visits = 10, .child_mean_value = -0.5F},
    };
    require(prunedRootPolicyVisits(winningAlternative, 100, 1.5F, true)[1] == 10,
            "Policy pruning removed a value-supported winning alternative");

    const std::vector<RootChildSearchStatistics> tied = {
        {.prior = 0.5F, .visits = 50, .child_mean_value = 0.0F},
        {.prior = 0.5F, .visits = 50, .child_mean_value = 0.0F},
    };
    const std::vector<std::uint32_t> tiedResult = prunedRootPolicyVisits(tied, 100, 1.5F, true);
    require(tiedResult[0] == 50 && tiedResult[1] == 50,
            "Deterministic visit tie changed an equally supported target");
}

} // namespace

int runGameSearchTreeTests() {
    try {
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        testForcedPlayoutMath();
        exercise_tree<ChessGame>(Board{});
        exercise_tree<Go7Game>(
            GoPosition<7>(GoRules{.komi_half_points = 15, .maximum_moves = 196}));
        exercise_tree<Go9Game>(
            GoPosition<9>(GoRules{.komi_half_points = 15, .maximum_moves = 324}));
        for (const float rootValue : {-0.8F, 0.8F}) {
            require(selectedRootEdgeAfterOneVisit(rootValue, 0.0F) != 0,
                    "Zero FPU reduction did not preserve the unvisited-child preference");
            require(selectedRootEdgeAfterOneVisit(rootValue, 0.2F) == 0,
                    "Reduced-parent FPU did not use the parent-value perspective");
            require(selectedRootEdgeAfterOneVisit(rootValue, 0.2F, 2.0F) != 0,
                    "Forced root playouts did not override ordinary FPU selection");
        }
        std::cout << "Shared chess and Go search-tree tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
