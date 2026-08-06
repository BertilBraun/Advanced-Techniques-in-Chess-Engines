#pragma once

#include "DirectInference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename Action> struct GameSearchEdge {
    Action action;
    float prior;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;
    std::optional<std::size_t> child_index;
};

template <typename Game> struct GameSearchNode {
    using Position = typename Game::Position;
    using Action = typename Game::Action;

    Position position;
    std::vector<GameSearchEdge<Action>> children;
    std::optional<std::size_t> parent_index;
    std::optional<std::size_t> parent_edge_index;
    std::uint32_t visits = 0;
    float value_sum = 0.0F;

    [[nodiscard]] bool expanded() const noexcept { return !children.empty(); }
};

template <typename Game> class GameSearchTree {
public:
    using Position = typename Game::Position;
    using Action = typename Game::Action;
    using Node = GameSearchNode<Game>;
    using Edge = GameSearchEdge<Action>;

    GameSearchTree(Position rootPosition, const std::size_t capacity)
        : m_capacity(capacity), m_initialPosition(rootPosition) {
        if (capacity == 0) {
            throw std::invalid_argument("Game search tree capacity must be positive");
        }
        m_nodes.reserve(capacity);
        m_nodes.push_back({std::move(rootPosition), {}, std::nullopt, std::nullopt});
    }

    [[nodiscard]] const Node &node(const std::size_t index) const { return m_nodes.at(index); }
    [[nodiscard]] Node &node(const std::size_t index) { return m_nodes.at(index); }
    [[nodiscard]] const Node &root() const { return node(m_rootIndex); }
    [[nodiscard]] Node &root() { return node(m_rootIndex); }
    [[nodiscard]] std::size_t rootIndex() const noexcept { return m_rootIndex; }
    [[nodiscard]] std::size_t liveNodeCount() const noexcept { return m_nodes.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

    [[nodiscard]] std::size_t selectLeaf(const float explorationConstant) {
        std::size_t selected = m_rootIndex;
        while (node(selected).expanded() && !Game::isTerminal(node(selected).position)) {
            const std::size_t edgeIndex = bestEdgeIndex(selected, explorationConstant);
            selected = materializeChild(selected, edgeIndex);
        }
        return selected;
    }

    void expand(const std::size_t nodeIndex, const std::vector<float> &policy) {
        Node &selected = node(nodeIndex);
        if (selected.expanded() || Game::isTerminal(selected.position)) {
            return;
        }
        const std::vector<Action> legalActions = Game::legalActions(selected.position);
        float total = 0.0F;
        for (const Action action : legalActions) {
            const int actionId = Game::actionId(action, selected.position);
            if (actionId < 0 || static_cast<std::size_t>(actionId) >= policy.size()) {
                throw std::invalid_argument("Game action lies outside the model policy");
            }
            total += std::max(0.0F, policy[static_cast<std::size_t>(actionId)]);
        }
        const float fallback = legalActions.empty() ? 0.0F : 1.0F / legalActions.size();
        selected.children.reserve(legalActions.size());
        for (const Action action : legalActions) {
            const float raw = std::max(
                0.0F, policy[static_cast<std::size_t>(Game::actionId(action, selected.position))]);
            selected.children.push_back({action, total > 0.0F ? raw / total : fallback});
        }
    }

    void backPropagate(std::size_t nodeIndex, float value) {
        while (true) {
            Node &selected = node(nodeIndex);
            ++selected.visits;
            selected.value_sum += value;
            if (!selected.parent_index.has_value()) {
                break;
            }
            Node &parent = node(*selected.parent_index);
            Edge &incoming = parent.children.at(*selected.parent_edge_index);
            ++incoming.visits;
            incoming.value_sum += value;
            value = -value;
            nodeIndex = *selected.parent_index;
        }
    }

    [[nodiscard]] std::size_t actionIdToEdgeIndex(const int actionId) const {
        const Node &selected = root();
        for (std::size_t index = 0; index < selected.children.size(); ++index) {
            if (Game::actionId(selected.children[index].action, selected.position) == actionId) {
                return index;
            }
        }
        throw std::invalid_argument("Selected action is not a root child");
    }

    void reroot(const int actionId) {
        const std::size_t edgeIndex = actionIdToEdgeIndex(actionId);
        const Position nextPosition =
            Game::childPosition(root().position, root().children[edgeIndex].action);
        m_initialPosition = nextPosition;
        reset();
    }

    void reset() {
        m_nodes.clear();
        m_rootIndex = 0;
        m_nodes.push_back({m_initialPosition, {}, std::nullopt, std::nullopt});
    }

private:
    std::size_t m_capacity;
    Position m_initialPosition;
    std::vector<Node> m_nodes;
    std::size_t m_rootIndex = 0;

    [[nodiscard]] std::size_t bestEdgeIndex(const std::size_t nodeIndex,
                                            const float explorationConstant) const {
        const Node &parent = node(nodeIndex);
        const float parentScale = std::sqrt(static_cast<float>(std::max(1U, parent.visits)));
        float bestScore = -std::numeric_limits<float>::infinity();
        std::size_t bestIndex = 0;
        for (std::size_t index = 0; index < parent.children.size(); ++index) {
            const Edge &edge = parent.children[index];
            const float meanValue = edge.visits == 0 ? 0.0F : edge.value_sum / edge.visits;
            const float exploration =
                explorationConstant * edge.prior * parentScale / (1.0F + edge.visits);
            const float score = -meanValue + exploration;
            if (score > bestScore) {
                bestScore = score;
                bestIndex = index;
            }
        }
        return bestIndex;
    }

    [[nodiscard]] std::size_t materializeChild(const std::size_t parentIndex,
                                               const std::size_t edgeIndex) {
        Edge &edge = node(parentIndex).children.at(edgeIndex);
        if (edge.child_index.has_value()) {
            return *edge.child_index;
        }
        if (m_nodes.size() == m_capacity) {
            throw std::overflow_error("Game search tree capacity exhausted");
        }
        const Position child = Game::childPosition(node(parentIndex).position, edge.action);
        const std::size_t childIndex = m_nodes.size();
        m_nodes.push_back({child, {}, parentIndex, edgeIndex});
        node(parentIndex).children[edgeIndex].child_index = childIndex;
        return childIndex;
    }
};

template <typename Game> class GameSearchRoot {
public:
    using Position = typename Game::Position;
    using Tree = GameSearchTree<Game>;

    GameSearchRoot(Position position, const std::size_t capacity)
        : m_tree(std::make_shared<Tree>(std::move(position), capacity)) {}

    [[nodiscard]] const Position &position() const { return m_tree->root().position; }
    [[nodiscard]] bool isTerminal() const { return Game::isTerminal(position()); }
    [[nodiscard]] std::uint32_t visits() const { return m_tree->root().visits; }
    [[nodiscard]] std::size_t liveNodeCount() const { return m_tree->liveNodeCount(); }
    [[nodiscard]] Tree &tree() { return *m_tree; }
    [[nodiscard]] const Tree &tree() const { return *m_tree; }
    [[nodiscard]] std::shared_ptr<Tree> sharedTree() const { return m_tree; }
    void play(const int actionId) { m_tree->reroot(actionId); }
    void reset() { m_tree->reset(); }

private:
    std::shared_ptr<Tree> m_tree;
};

struct GameSearchVisit {
    int action_id;
    std::uint32_t visit_count;
};

struct GameSearchResult {
    float root_value;
    std::vector<GameSearchVisit> visits;
};

template <typename Game> class BatchedGameSearch {
public:
    using Root = GameSearchRoot<Game>;
    using Tree = GameSearchTree<Game>;

    BatchedGameSearch(const std::string &modelPath, const InferenceDevice device,
                      const int deviceId, const std::size_t maximumBatchSize,
                      const InferenceDimensions dimensions, const float explorationConstant,
                      const std::size_t treeCapacity, const std::uint64_t modelGeneration)
        : m_runner(modelPath, device, deviceId, maximumBatchSize, false, dimensions),
          m_dimensions(dimensions), m_explorationConstant(explorationConstant),
          m_treeCapacity(treeCapacity), m_modelGeneration(modelGeneration) {}

    [[nodiscard]] Root newRoot(typename Game::Position position) {
        Root root(std::move(position), m_treeCapacity);
        m_trees.push_back(root.sharedTree());
        return root;
    }

    [[nodiscard]] std::vector<GameSearchResult> search(std::vector<Root> &roots,
                                                       const std::uint32_t simulations) {
        if (roots.empty() || roots.size() > m_runner.maximumBatchSize()) {
            throw std::invalid_argument("Game search root batch is outside inference capacity");
        }
        torch::Tensor inputs = m_runner.createInputBuffer();
        DirectInferenceOutput outputs = m_runner.createOutputBuffer();
        for (std::uint32_t simulation = 0; simulation < simulations; ++simulation) {
            std::vector<std::pair<std::size_t, std::size_t>> pending;
            for (std::size_t rootIndex = 0; rootIndex < roots.size(); ++rootIndex) {
                Tree &tree = roots[rootIndex].tree();
                const std::size_t leafIndex = tree.selectLeaf(m_explorationConstant);
                const auto terminalValue = Game::terminalValue(tree.node(leafIndex).position);
                if (Game::isTerminal(tree.node(leafIndex).position)) {
                    tree.backPropagate(leafIndex, terminalValue.value_or(0.0F));
                } else {
                    pending.emplace_back(rootIndex, leafIndex);
                }
            }
            if (pending.empty()) {
                continue;
            }
            for (std::size_t row = 0; row < pending.size(); ++row) {
                const auto [rootIndex, leafIndex] = pending[row];
                std::int8_t *destination = inputs[row].data_ptr<std::int8_t>();
                Game::encodeInputInto(roots[rootIndex].tree().node(leafIndex).position,
                                      destination);
            }
            m_runner.forwardInto(inputs, pending.size(), outputs);
            for (std::size_t row = 0; row < pending.size(); ++row) {
                const auto [rootIndex, leafIndex] = pending[row];
                const torch::Tensor policyRow = outputs.policies[row];
                std::vector<float> policy(static_cast<std::size_t>(m_dimensions.actions));
                std::copy_n(policyRow.data_ptr<float>(), policy.size(), policy.begin());
                Tree &tree = roots[rootIndex].tree();
                tree.expand(leafIndex, policy);
                const torch::Tensor outcomeRow = outputs.outcomes[row];
                const float value = outcomeRow[0].item<float>() - outcomeRow[2].item<float>();
                tree.backPropagate(leafIndex, value);
            }
        }
        std::vector<GameSearchResult> results;
        results.reserve(roots.size());
        for (const Root &root : roots) {
            const auto &node = root.tree().root();
            GameSearchResult result{node.visits == 0 ? 0.0F : node.value_sum / node.visits, {}};
            result.visits.reserve(node.children.size());
            for (const auto &edge : node.children) {
                result.visits.push_back({Game::actionId(edge.action, node.position), edge.visits});
            }
            results.push_back(std::move(result));
        }
        return results;
    }

    void refreshModel(const std::uint64_t modelGeneration, const std::string &modelPath) {
        if (modelGeneration <= m_modelGeneration) {
            throw std::invalid_argument("Model generation must increase during refresh");
        }
        PreparedInferenceModel prepared = m_runner.prepareModelRefresh(modelPath);
        m_runner.commitModelRefresh(std::move(prepared));
        m_modelGeneration = modelGeneration;
        for (const std::weak_ptr<Tree> &tree : m_trees) {
            if (const std::shared_ptr<Tree> active = tree.lock()) {
                active->reset();
            }
        }
    }

    [[nodiscard]] std::uint64_t modelGeneration() const noexcept { return m_modelGeneration; }

private:
    DirectInferenceRunner m_runner;
    InferenceDimensions m_dimensions;
    float m_explorationConstant;
    std::size_t m_treeCapacity;
    std::uint64_t m_modelGeneration;
    std::vector<std::weak_ptr<Tree>> m_trees;
};
