#include "games/chess/ChessGameContract.hpp"
#include "games/go/GoGameContract.hpp"
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

template <typename Game> void exercise_tree(typename Game::Position position) {
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
    const int selectedAction = Game::actionId(
        tree.root().children[*leafNode.parent_edge_index].action, tree.root().position);
    tree.reroot(selectedAction);
    require(tree.root().visits == 1, "Shared game tree discarded retained child statistics");
    require(tree.liveNodeCount() == 1, "Shared game tree did not reclaim the previous root");
    tree.reset();
    require(tree.root().visits == 0, "Shared game tree reset retained old statistics");
}

} // namespace

int main() {
    try {
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        exercise_tree<ChessGameContract>(ChessGameContract::initialPosition());
        exercise_tree<Go7GameContract>(Go7GameContract::initialPosition(GoRules{15, 196}));
        exercise_tree<Go9GameContract>(Go9GameContract::initialPosition(GoRules{15, 324}));
        std::cout << "Shared chess and Go search-tree tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
