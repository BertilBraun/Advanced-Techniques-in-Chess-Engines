#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "games/go/GoGame.hpp"
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

std::size_t selectedRootEdgeAfterOneVisit(const float rootValue, const float fpuReduction) {
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

    const std::size_t selectedLeaf = *tree.selectAvailableLeaf(0.1F, fpuReduction);
    return *tree.node(selectedLeaf).parent_edge_index;
}

} // namespace

int runGameSearchTreeTests() {
    try {
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
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
        }
        std::cout << "Shared chess and Go search-tree tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
