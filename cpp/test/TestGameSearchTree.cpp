#include "MCTS/GameSearch.hpp"
#include "games/chess/ChessGameContract.hpp"
#include "games/go/GoGameContract.hpp"

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

template <typename Game>
void exercise_tree(typename Game::Position position, const std::size_t actionCount) {
    GameSearchTree<Game> tree(std::move(position), 16);
    tree.expand(tree.rootIndex(), std::vector<float>(actionCount, 1.0F));
    require(tree.root().children.size() == Game::legalActions(tree.root().position).size(),
            "Shared game tree did not create every legal edge");
    const std::size_t leaf = tree.selectLeaf(1.5F);
    require(leaf != tree.rootIndex(), "Shared game tree did not materialize a child");
    tree.backPropagate(leaf, 0.5F);
    require(tree.root().visits == 1, "Shared game tree did not backpropagate to the root");
    require(tree.liveNodeCount() == 2, "Shared game tree created an unexpected node count");
}

} // namespace

int main() {
    try {
        Bitboards::init();
        Position::init();
        exercise_tree<ChessGameContract>(ChessGameContract::initialPosition(), ACTION_SIZE);
        exercise_tree<Go7GameContract>(Go7GameContract::initialPosition(GoRules{15, 196}), 50);
        exercise_tree<Go9GameContract>(Go9GameContract::initialPosition(GoRules{15, 324}), 82);
        std::cout << "Shared chess and Go search-tree tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
