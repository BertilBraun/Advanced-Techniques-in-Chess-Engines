#include "TestRunner.hpp"
#include "games/chess/ChessGame.hpp"
#include "games/go/GoGame.hpp"
#include "search/ForcedPlayouts.hpp"
#include "search/SearchTree.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(const float actual, const float expected, const char *message) {
    if (std::abs(actual - expected) > 1e-5F) {
        throw std::runtime_error(message);
    }
}

using GoTree = GameSearchTree<Go7Game>;

GoTree makeGoTree(const std::size_t capacity, const float turnDiscount = 1.0F) {
    return GoTree(GoPosition<7>(GoRules{.komi_half_points = 15, .maximum_moves = 196}), capacity,
                  capacity, turnDiscount);
}

void expandRoot(GoTree &tree, const std::size_t edgeCount) {
    const std::vector<Go7Game::Action> legalActions = Go7Game::legalActions(tree.root().position);
    require(legalActions.size() >= edgeCount, "Go fixture does not have enough root actions");
    SearchInferenceResult<Go7Game> inference{{}, {0.5F, 0.0F, 0.5F}};
    for (const auto index : range(edgeCount)) {
        inference.actions.emplace_back(legalActions[index], 1.0F / static_cast<float>(edgeCount));
    }
    tree.expand(tree.rootIndex(), inference);
}

std::size_t materializeRootEdge(GoTree &tree, const std::size_t selectedEdge) {
    for (const auto index : range(tree.root().children.size())) {
        tree.root().children[index].prior = index == selectedEdge ? 1.0F : 0.0F;
    }
    const std::optional<std::size_t> selected = tree.selectAvailableLeaf(1000.0F);
    require(selected.has_value(), "Go fixture could not materialize a root edge");
    require(tree.node(*selected).parent_index == tree.rootIndex(),
            "Go fixture selected below the root");
    require(tree.node(*selected).parent_edge_index == selectedEdge,
            "Go fixture materialized the wrong root edge");
    return *selected;
}

void requireConsistentMaterializedStatistics(const GoTree &tree, const std::size_t nodeIndex) {
    const GameSearchNode<Go7Game> &selected = tree.node(nodeIndex);
    std::uint64_t childVisits = 0;
    for (const GameSearchEdge<Go7Game::Action> &edge : selected.children) {
        childVisits += edge.visits;
        if (!edge.child_index.has_value()) {
            continue;
        }
        const GameSearchNode<Go7Game> &child = tree.node(*edge.child_index);
        require(child.visits == edge.visits,
                "Materialized child visits disagree with the incoming edge");
        requireNear(child.value_sum, edge.value_sum,
                    "Materialized child value disagrees with the incoming edge");
        requireConsistentMaterializedStatistics(tree, *edge.child_index);
    }
    require(childVisits <= selected.visits, "Child visits exceed their parent visits");
}

void testConsistentDiscount() {
    GoTree tree = makeGoTree(8, 0.5F);
    expandRoot(tree, 3);
    const std::size_t firstChild = materializeRootEdge(tree, 0);
    const std::size_t secondChild = materializeRootEdge(tree, 1);

    tree.root().visits = 10;
    tree.root().value_sum = 0.65F;
    tree.root().children[0].visits = 5;
    tree.root().children[0].value_sum = 1.0F;
    tree.node(firstChild).visits = 5;
    tree.node(firstChild).value_sum = 1.0F;
    tree.root().children[1].visits = 3;
    tree.root().children[1].value_sum = -1.5F;
    tree.node(secondChild).visits = 3;
    tree.node(secondChild).value_sum = -1.5F;

    tree.discount(0.6F);

    require(tree.root().visits == 6, "Discount did not retain the exact root visit mass");
    require(tree.root().children[0].visits == 3 && tree.root().children[1].visits == 2,
            "Discount did not use largest-remainder child allocation");
    require(tree.node(firstChild).visits == 3 && tree.node(secondChild).visits == 2,
            "Discount did not propagate retained edge visits into child nodes");
    requireNear(tree.node(firstChild).value_sum, 0.6F,
                "Discount changed the first retained child mean");
    requireNear(tree.node(secondChild).value_sum, -1.0F,
                "Discount changed the second retained child mean");
    requireNear(tree.root().value_sum, 0.4F,
                "Discount did not reconstruct the parent value from local and child buckets");
    requireConsistentMaterializedStatistics(tree, tree.rootIndex());
}

void testLocalAndZeroVisitRetention() {
    GoTree localTree = makeGoTree(4);
    localTree.root().visits = 5;
    localTree.root().value_sum = 2.5F;
    localTree.discount(0.6F);
    require(localTree.root().visits == 3, "Local-only discount retained the wrong visits");
    requireNear(localTree.root().value_sum, 1.5F, "Local-only discount changed the retained mean");

    GoTree zeroTree = makeGoTree(4);
    expandRoot(zeroTree, 1);
    const std::size_t child = materializeRootEdge(zeroTree, 0);
    zeroTree.root().visits = 1;
    zeroTree.root().value_sum = -0.25F;
    zeroTree.root().children[0].visits = 1;
    zeroTree.root().children[0].value_sum = 0.25F;
    zeroTree.node(child).visits = 1;
    zeroTree.node(child).value_sum = 0.25F;
    zeroTree.discount(0.0F);
    require(zeroTree.root().visits == 0 && zeroTree.liveNodeCount() == 1,
            "Zero retention did not reclaim the complete materialized subtree");
    require(!zeroTree.root().children[0].child_index.has_value() &&
                zeroTree.root().children[0].visits == 0,
            "Zero retention left child statistics attached");
}

void testRecursiveDiscount() {
    GoTree tree = makeGoTree(8);
    expandRoot(tree, 1);
    const std::size_t childIndex = materializeRootEdge(tree, 0);
    const std::vector<Go7Game::Action> legalActions =
        Go7Game::legalActions(tree.node(childIndex).position);
    SearchInferenceResult<Go7Game> inference{{}, {0.5F, 0.0F, 0.5F}};
    inference.actions.emplace_back(legalActions[0], 0.5F);
    inference.actions.emplace_back(legalActions[1], 0.5F);
    tree.expand(childIndex, inference);

    tree.node(childIndex).children[0].prior = 1.0F;
    tree.node(childIndex).children[1].prior = 0.0F;
    const std::size_t firstGrandchild = *tree.selectAvailableLeaf(1000.0F);
    tree.node(childIndex).children[0].prior = 0.0F;
    tree.node(childIndex).children[1].prior = 1.0F;
    const std::size_t secondGrandchild = *tree.selectAvailableLeaf(1000.0F);

    tree.root().visits = 9;
    tree.root().value_sum = -0.9F;
    tree.root().children[0].visits = 7;
    tree.root().children[0].value_sum = 0.5F;
    tree.node(childIndex).visits = 7;
    tree.node(childIndex).value_sum = 0.5F;
    tree.node(childIndex).children[0].visits = 4;
    tree.node(childIndex).children[0].value_sum = 0.8F;
    tree.node(firstGrandchild).visits = 4;
    tree.node(firstGrandchild).value_sum = 0.8F;
    tree.node(childIndex).children[1].visits = 2;
    tree.node(childIndex).children[1].value_sum = -1.0F;
    tree.node(secondGrandchild).visits = 2;
    tree.node(secondGrandchild).value_sum = -1.0F;

    tree.discount(0.6F);

    require(tree.root().visits == 5 && tree.node(childIndex).visits == 4,
            "Recursive discount did not propagate the top-down visit allocation");
    require(tree.node(firstGrandchild).visits == 2 && tree.node(secondGrandchild).visits == 1,
            "Recursive discount did not allocate nested visits consistently");
    requireNear(tree.node(childIndex).value_sum, 0.4F,
                "Recursive discount did not reconstruct the nested value");
    requireNear(tree.root().value_sum, -0.6F,
                "Recursive discount did not reconstruct the root value");
    requireConsistentMaterializedStatistics(tree, tree.rootIndex());
}

void testImportancePruningAndRematerialization() {
    GoTree tree = makeGoTree(5);
    expandRoot(tree, 3);
    const std::size_t firstChild = materializeRootEdge(tree, 0);
    const std::size_t secondChild = materializeRootEdge(tree, 1);
    const std::size_t thirdChild = materializeRootEdge(tree, 2);
    const std::array visits = {10U, 5U, 1U};
    const std::array childIndices = {firstChild, secondChild, thirdChild};
    tree.root().visits = 16;
    tree.root().value_sum = 0.0F;
    for (const auto index : range(visits.size())) {
        tree.root().children[index].visits = visits[index];
        tree.root().children[index].value_sum = 0.0F;
        tree.node(childIndices[index]).visits = visits[index];
        tree.node(childIndices[index]).value_sum = 0.0F;
    }

    tree.prepareForSearch(16, 1);

    require(tree.root().children[0].child_index.has_value() &&
                tree.root().children[1].child_index.has_value(),
            "Arena pruning discarded a more-visited child");
    require(!tree.root().children[2].child_index.has_value(),
            "Arena pruning did not discard the least-visited leaf");
    require(tree.root().children[2].visits == 1,
            "Arena pruning discarded the summarized edge statistics");

    const std::size_t rematerialized = materializeRootEdge(tree, 2);
    require(tree.node(rematerialized).visits == 1,
            "Rematerialized child did not inherit summarized visits");
    requireNear(tree.node(rematerialized).value_sum, tree.root().children[2].value_sum,
                "Rematerialized child did not inherit summarized value");
    requireConsistentMaterializedStatistics(tree, tree.rootIndex());
}

void testFreshRootNoiseUsesRawPriors() {
    GoTree twiceNoised = makeGoTree(4);
    GoTree onceNoised = makeGoTree(4);
    GoTree randomAdvance = makeGoTree(4);
    expandRoot(twiceNoised, 3);
    expandRoot(onceNoised, 3);
    expandRoot(randomAdvance, 3);
    std::mt19937 firstRandom(17);
    std::mt19937 secondRandom(17);
    twiceNoised.addRootNoise(0.3F, 0.25F, firstRandom);
    randomAdvance.addRootNoise(0.3F, 0.25F, secondRandom);
    twiceNoised.addRootNoise(0.3F, 0.25F, firstRandom);
    onceNoised.addRootNoise(0.3F, 0.25F, secondRandom);
    for (const auto index : range(twiceNoised.root().children.size())) {
        requireNear(twiceNoised.root().children[index].prior,
                    onceNoised.root().children[index].prior,
                    "Fresh root noise compounded a previous noisy prior");
        requireNear(twiceNoised.root().children[index].raw_prior, 1.0F / 3.0F,
                    "Root noise mutated the raw network prior");
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
        testConsistentDiscount();
        testLocalAndZeroVisitRetention();
        testRecursiveDiscount();
        testImportancePruningAndRematerialization();
        testFreshRootNoiseUsesRawPriors();
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
