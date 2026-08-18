#include "TestRunner.hpp"
#include "search/SearchTree.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

enum class FixtureKind : std::uint8_t { TerminalDiamond, OpenDiamond, DeepDiamond, Cycle };

struct FixtureState {
    int id;
    FixtureKind kind;
    int history_token;

    [[nodiscard]] bool operator==(const FixtureState &) const noexcept = default;
};

struct FixtureEncoding {
    [[nodiscard]] static constexpr InferenceDimensions inferenceDimensions() {
        return {.channels = 1, .rows = 1, .columns = 1, .actions = 2, .outcomes = 3};
    }
    [[nodiscard]] static int actionId(const int action, const FixtureState &) { return action; }
    static void encodeInputInto(const FixtureState &state, std::int8_t *destination) {
        *destination = static_cast<std::int8_t>(state.id);
    }
};

struct FixtureGame {
    using State = FixtureState;
    using Action = int;
    using Encoding = FixtureEncoding;

    [[nodiscard]] static State childState(const State &parent, const Action action) {
        if (parent.kind == FixtureKind::Cycle) {
            return {.id = parent.id == 0 ? 1 : 0,
                    .kind = parent.kind,
                    .history_token = parent.history_token};
        }
        if (parent.id == 0) {
            return {.id = action == 0 ? 1 : 2,
                    .kind = parent.kind,
                    .history_token = parent.history_token};
        }
        if (parent.kind == FixtureKind::DeepDiamond && parent.id == 3) {
            return {.id = 4, .kind = parent.kind, .history_token = parent.history_token};
        }
        return {.id = 3, .kind = parent.kind, .history_token = parent.history_token};
    }
    [[nodiscard]] static std::vector<Action> legalActions(const State &state) {
        if (state.kind == FixtureKind::Cycle) {
            return {0};
        }
        if (state.id == 0) {
            return {0, 1};
        }
        if (state.id == 1 || state.id == 2) {
            return {0};
        }
        if (state.kind == FixtureKind::DeepDiamond && state.id == 3) {
            return {0};
        }
        return {};
    }
    [[nodiscard]] static bool isTerminal(const State &state) {
        return (state.kind == FixtureKind::TerminalDiamond && state.id == 3) ||
               (state.kind == FixtureKind::DeepDiamond && state.id == 4);
    }
    [[nodiscard]] static float terminalValue(const State &) { return 0.25F; }
    [[nodiscard]] static float cycleValue(const State &) { return 0.0F; }
    [[nodiscard]] static std::size_t stateHash(const State &state) noexcept {
        return static_cast<std::size_t>(state.id * 31 + static_cast<int>(state.kind) * 7 +
                                        state.history_token * 101);
    }
    [[nodiscard]] static bool statesEqual(const State &left, const State &right) noexcept {
        return left == right;
    }
};

static_assert(SearchGame<FixtureGame>);

using FixtureTree = GameSearchTree<FixtureGame>;

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

[[nodiscard]] TreeSearchParameters graphParameters(const bool transpositionsEnabled = true) {
    return TreeSearchParameters(
        1000.0F, FirstPlayUrgencyParameters(FirstPlayUrgencyKind::Zero), 0.0F, 1.0F,
        MonteCarloGraphSearchParameters(0.01F, transpositionsEnabled));
}

[[nodiscard]] FixtureTree makeGraph(const FixtureKind kind,
                                    const bool transpositionsEnabled = true) {
    return FixtureTree({.id = 0, .kind = kind, .history_token = 0}, 16, 32, 1.0F,
                       MonteCarloGraphSearchParameters(0.01F, transpositionsEnabled));
}

void expand(FixtureTree &tree, const std::size_t nodeIndex) {
    const auto actions = FixtureGame::legalActions(tree.node(nodeIndex).position);
    SearchInferenceResult<FixtureGame> inference{{}, {.win = 0.5F, .draw = 0.0F, .loss = 0.5F}};
    for (const int action : actions) {
        inference.actions.emplace_back(action, 1.0F / static_cast<float>(actions.size()));
    }
    tree.expand(nodeIndex, inference);
}

void preferRootEdge(FixtureTree &tree, const std::size_t edgeIndex) {
    for (std::size_t index = 0; index < tree.root().children.size(); ++index) {
        tree.root().children[index].prior = index == edgeIndex ? 1.0F : 0.0F;
    }
}

[[nodiscard]] GraphSearchSelection selectGraph(FixtureTree &tree) {
    const std::optional<GraphSearchSelection> selected =
        tree.selectAvailableGraphLeaf(graphParameters());
    require(selected.has_value(), "Graph fixture did not find an available selection");
    return *selected;
}

void materializeDiamond(FixtureTree &tree) {
    expand(tree, tree.rootIndex());
    preferRootEdge(tree, 0);
    GraphSearchSelection firstParent = selectGraph(tree);
    expand(tree, firstParent.path.leaf_index);
    tree.backPropagateGraph(firstParent.path, 0.0F, true);
    GraphSearchSelection firstTerminal = selectGraph(tree);
    require(firstTerminal.immediate_value.has_value(), "Diamond terminal was not detected");
    tree.backPropagateGraph(firstTerminal.path, *firstTerminal.immediate_value,
                            firstTerminal.update_leaf);

    preferRootEdge(tree, 1);
    GraphSearchSelection secondParent = selectGraph(tree);
    expand(tree, secondParent.path.leaf_index);
    tree.backPropagateGraph(secondParent.path, 0.0F, true);
    GraphSearchSelection secondTerminal = selectGraph(tree);
    require(secondTerminal.immediate_value.has_value(), "Shared terminal was not detected");
    tree.backPropagateGraph(secondTerminal.path, *secondTerminal.immediate_value,
                            secondTerminal.update_leaf);
}

void testSharedNodesAndCorrection() {
    FixtureTree tree = makeGraph(FixtureKind::TerminalDiamond);
    materializeDiamond(tree);
    const std::size_t firstParent = *tree.root().children[0].child_index;
    const std::size_t secondParent = *tree.root().children[1].child_index;
    const std::size_t shared = *tree.node(firstParent).children[0].child_index;
    require(tree.node(secondParent).children[0].child_index == shared,
            "Diamond paths did not share one canonical node");
    require(tree.node(shared).incoming_edges == 2,
            "Shared graph node did not record both incoming edges");
    require(tree.liveNodeCount() == 4, "Diamond graph allocated a duplicate state node");
    require(tree.graphStatistics().transposition_links == 1,
            "Diamond transposition link was not instrumented");

    auto &incoming = tree.node(secondParent).children[0];
    tree.node(shared).visits = 10;
    tree.node(shared).value_sum = 5.0F;
    incoming.visits = 2;
    incoming.value_sum = 0.8F;
    const GraphSearchStatistics statisticsBefore = tree.graphStatistics();
    preferRootEdge(tree, 1);
    const GraphSearchSelection correction = selectGraph(tree);
    require(correction.immediate_value.has_value() && !correction.update_leaf,
            "Stale incoming edge did not produce a graph correction");
    tree.backPropagateGraph(correction.path, *correction.immediate_value,
                            correction.update_leaf);
    requireNear(incoming.value_sum / static_cast<float>(incoming.visits), 0.5F,
                "Graph correction did not move the edge mean to the shared-node target");
    require(tree.node(shared).visits == 10,
            "Correction-only backup incorrectly revisited the shared node");
    require(tree.graphStatistics().transposition_traversals ==
                    statisticsBefore.transposition_traversals + 1 &&
                tree.graphStatistics().shared_node_visits_observed ==
                    statisticsBefore.shared_node_visits_observed + 10 &&
                tree.graphStatistics().incoming_edge_visits_observed ==
                    statisticsBefore.incoming_edge_visits_observed + 2 &&
                tree.graphStatistics().shared_visit_advantage ==
                    statisticsBefore.shared_visit_advantage + 8 &&
                tree.graphStatistics().maximum_shared_visit_advantage >= 8,
            "Shared-node visit advantage instrumentation is incorrect");

    tree.node(shared).visits = 10;
    tree.node(shared).value_sum = 10.0F;
    incoming.visits = 5;
    incoming.value_sum = -5.0F;
    const std::uint64_t clipsBefore = tree.graphStatistics().correction_clips;
    const GraphSearchSelection clippedCorrection = selectGraph(tree);
    tree.backPropagateGraph(clippedCorrection.path, *clippedCorrection.immediate_value,
                            clippedCorrection.update_leaf);
    requireNear(*clippedCorrection.immediate_value, 1.0F,
                "Out-of-range graph correction was not clipped to the game value range");
    require(std::abs(incoming.value_sum / static_cast<float>(incoming.visits) - 1.0F) < 2.0F,
            "Clipped graph correction did not move monotonically toward the shared-node target");
    require(tree.graphStatistics().correction_clips == clipsBefore + 1,
            "Clipped graph correction was not instrumented");
}

void testUnfoldedTreeCountsSharedDescendants() {
    FixtureTree tree = makeGraph(FixtureKind::DeepDiamond);
    expand(tree, tree.rootIndex());

    preferRootEdge(tree, 0);
    GraphSearchSelection firstParent = selectGraph(tree);
    expand(tree, firstParent.path.leaf_index);
    tree.backPropagateGraph(firstParent.path, 0.0F, true);
    GraphSearchSelection shared = selectGraph(tree);
    expand(tree, shared.path.leaf_index);
    tree.backPropagateGraph(shared.path, 0.0F, true);
    GraphSearchSelection descendant = selectGraph(tree);
    tree.backPropagateGraph(descendant.path, *descendant.immediate_value,
                            descendant.update_leaf);

    preferRootEdge(tree, 1);
    GraphSearchSelection secondParent = selectGraph(tree);
    expand(tree, secondParent.path.leaf_index);
    tree.backPropagateGraph(secondParent.path, 0.0F, true);
    static_cast<void>(selectGraph(tree));

    const GraphStructureStatistics structure = tree.graphStructureStatistics();
    require(structure.canonical_nodes == 5 && structure.materialized_edges == 5 &&
                structure.expanded_nodes == 4 && structure.shared_nodes == 1,
            "Canonical graph structure counts are incorrect");
    require(structure.unfolded_tree_nodes == 7 && structure.unfolded_tree_edges == 6 &&
                structure.unfolded_expanded_nodes == 5 &&
                structure.maximum_path_multiplicity == 2 && !structure.saturated,
            "Unfolded tree did not duplicate the shared node and its descendant");
}

void testFirstTranspositionReusesSharedEvaluation() {
    FixtureTree tree = makeGraph(FixtureKind::DeepDiamond);
    expand(tree, tree.rootIndex());

    preferRootEdge(tree, 0);
    GraphSearchSelection firstParent = selectGraph(tree);
    expand(tree, firstParent.path.leaf_index);
    tree.backPropagateGraph(firstParent.path, 0.0F, true);
    GraphSearchSelection shared = selectGraph(tree);
    expand(tree, shared.path.leaf_index);
    tree.backPropagateGraph(shared.path, 0.4F, true);

    preferRootEdge(tree, 1);
    GraphSearchSelection secondParent = selectGraph(tree);
    expand(tree, secondParent.path.leaf_index);
    tree.backPropagateGraph(secondParent.path, 0.0F, true);
    const GraphSearchStatistics statisticsBefore = tree.graphStatistics();
    const GraphSearchSelection reused = selectGraph(tree);

    require(reused.immediate_value.has_value() && !reused.update_leaf,
            "First transposition link did not reuse the shared node evaluation");
    requireNear(*reused.immediate_value, 0.4F,
                "First transposition link backed up the wrong shared value");
    require(tree.graphStatistics().evaluations_avoided ==
                statisticsBefore.evaluations_avoided + 1,
            "First transposition evaluation reuse was not instrumented");

    tree.backPropagateGraph(reused.path, *reused.immediate_value, reused.update_leaf);
    const GameSearchPathStep &incomingStep = reused.path.steps.back();
    const auto &incoming =
        tree.node(incomingStep.node_index).children.at(incomingStep.edge_index);
    require(incoming.visits == 1,
            "First transposition reuse did not visit the new incoming edge exactly once");
    requireNear(incoming.value_sum, 0.4F,
                "First transposition reuse did not initialize the incoming edge from the node");
    require(tree.node(shared.path.leaf_index).visits == 1,
            "First transposition reuse incorrectly revisited the shared node");
}

void testHistoryDistinctStatesDoNotMerge() {
    FixtureTree tree = makeGraph(FixtureKind::TerminalDiamond);
    expand(tree, tree.rootIndex());
    FixtureState left = FixtureGame::childState(tree.root().position, 0);
    FixtureState right = left;
    right.history_token = 1;
    require(FixtureGame::stateHash(left) != FixtureGame::stateHash(right) &&
                !FixtureGame::statesEqual(left, right),
            "Fixture history must participate in semantic graph identity");
}

void testCycleCutoff() {
    FixtureTree tree = makeGraph(FixtureKind::Cycle);
    expand(tree, tree.rootIndex());
    GraphSearchSelection child = selectGraph(tree);
    expand(tree, child.path.leaf_index);
    tree.backPropagateGraph(child.path, 0.0F, true);
    const GraphSearchSelection cycle = selectGraph(tree);
    require(cycle.immediate_value == 0.0F && !cycle.update_leaf,
            "Cycle was not converted to the game-owned cycle value");
    tree.backPropagateGraph(cycle.path, *cycle.immediate_value, cycle.update_leaf);
    require(tree.liveNodeCount() == 2 && tree.graphStatistics().cycle_cutoffs == 1,
            "Cycle handling linked a back-edge or missed instrumentation");
    require(!tree.node(child.path.leaf_index).children[0].child_index.has_value(),
            "Cycle handling retained a graph back-edge");
}

void testParallelReservationCoalescesLeaf() {
    FixtureTree tree = makeGraph(FixtureKind::OpenDiamond);
    expand(tree, tree.rootIndex());
    preferRootEdge(tree, 0);
    GraphSearchSelection firstParent = selectGraph(tree);
    expand(tree, firstParent.path.leaf_index);
    tree.backPropagateGraph(firstParent.path, 0.0F, true);
    GraphSearchSelection sharedLeaf = selectGraph(tree);
    tree.reserveGraph(sharedLeaf.path);

    preferRootEdge(tree, 1);
    GraphSearchSelection secondParent = selectGraph(tree);
    expand(tree, secondParent.path.leaf_index);
    tree.backPropagateGraph(secondParent.path, 0.0F, true);
    const std::optional<GraphSearchSelection> duplicate =
        tree.selectAvailableGraphLeaf(graphParameters());
    require(!duplicate.has_value(),
            "Parallel graph visit scheduled duplicate inference for one canonical leaf");
    tree.cancelGraphReservation(sharedLeaf.path);
    require(!tree.node(sharedLeaf.path.leaf_index).inference_pending,
            "Cancelled graph reservation left inference pending");
    requireNear(tree.root().virtual_loss, 0.0F,
                "Cancelled parallel graph visit left root virtual loss");
}

void testRerootMarkAndSweep() {
    FixtureTree tree = makeGraph(FixtureKind::TerminalDiamond);
    materializeDiamond(tree);
    tree.rerootEdge(0);
    require(tree.root().position.id == 1, "Graph reroot selected the wrong child");
    require(tree.liveNodeCount() == 2, "Graph reroot did not reclaim unreachable nodes");
    require(tree.node(*tree.root().children[0].child_index).position.id == 3,
            "Graph reroot failed to retain the reachable shared descendant");
    require(tree.graphStatistics().nodes_reclaimed == 2,
            "Graph reroot reclamation instrumentation is incorrect");
}

void testSharedLeafPruningDetachesEveryParent() {
    FixtureTree tree({.id = 0, .kind = FixtureKind::TerminalDiamond, .history_token = 0}, 8, 8,
                     1.0F, MonteCarloGraphSearchParameters{});
    materializeDiamond(tree);
    const std::size_t firstParent = *tree.root().children[0].child_index;
    const std::size_t secondParent = *tree.root().children[1].child_index;

    tree.prepareForSearch(tree.root().visits + 4, 1);

    require(tree.liveNodeCount() == 3, "Graph capacity pruning retained a shared leaf");
    require(!tree.node(firstParent).children[0].child_index.has_value() &&
                !tree.node(secondParent).children[0].child_index.has_value(),
            "Graph capacity pruning left an incoming edge to a reclaimed shared node");
    require(tree.graphStatistics().nodes_pruned == 1,
            "Graph capacity pruning instrumentation is incorrect");
}

void testRetainedStatisticsScaleSharedNodeOnce() {
    FixtureTree tree = makeGraph(FixtureKind::TerminalDiamond);
    materializeDiamond(tree);
    const std::size_t firstParent = *tree.root().children[0].child_index;
    const std::size_t shared = *tree.node(firstParent).children[0].child_index;
    const float sharedMean =
        tree.node(shared).value_sum / static_cast<float>(tree.node(shared).visits);

    tree.discount(0.75F);

    require(tree.root().visits == 3, "Graph retained-root discount scaled the root incorrectly");
    require(tree.node(shared).visits == 1,
            "Graph retained-root discount scaled a shared node once per parent");
    requireNear(tree.node(shared).value_sum, sharedMean,
                "Graph retained-root discount changed the shared-node mean");
}

void testNoTranspositionTreeEquivalence() {
    FixtureTree tree({.id = 0, .kind = FixtureKind::TerminalDiamond, .history_token = 0}, 16,
                     32);
    FixtureTree graph = makeGraph(FixtureKind::TerminalDiamond, false);
    expand(tree, tree.rootIndex());
    expand(graph, graph.rootIndex());
    tree.root().children[0].prior = 1.0F;
    tree.root().children[1].prior = 0.0F;
    preferRootEdge(graph, 0);
    const std::size_t treeLeaf = *tree.selectAvailableLeaf(graphParameters(false));
    const GraphSearchSelection graphLeaf = selectGraph(graph);
    tree.backPropagate(treeLeaf, 0.4F);
    graph.backPropagateGraph(graphLeaf.path, 0.4F, true);
    require(tree.root().visits == graph.root().visits,
            "No-transposition graph changed root visits");
    requireNear(tree.root().value_sum, graph.root().value_sum,
                "No-transposition graph changed root value backup");
    require(tree.root().children[0].visits == graph.root().children[0].visits,
            "No-transposition graph changed edge visits");
    requireNear(tree.root().children[0].value_sum, graph.root().children[0].value_sum,
                "No-transposition graph changed edge values");
}

} // namespace

int runMonteCarloGraphSearchTests() {
    try {
        testSharedNodesAndCorrection();
        testUnfoldedTreeCountsSharedDescendants();
        testFirstTranspositionReusesSharedEvaluation();
        testHistoryDistinctStatesDoNotMerge();
        testCycleCutoff();
        testParallelReservationCoalescesLeaf();
        testRerootMarkAndSweep();
        testSharedLeafPruningDetachesEveryParent();
        testRetainedStatisticsScaleSharedNodeOnce();
        testNoTranspositionTreeEquivalence();
        std::cout << "Monte-Carlo graph-search tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
