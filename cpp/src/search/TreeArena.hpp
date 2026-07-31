#pragma once

#include "common.hpp"

#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace az::search {

namespace detail {
struct NodeIndexTag;
struct EdgeIndexTag;
} // namespace detail

template <typename Tag> class ArenaIndex {
public:
    [[nodiscard]] bool operator==(const ArenaIndex &) const = default;

private:
    explicit ArenaIndex(uint64 packed) : _packed(packed) {}

    [[nodiscard]] uint32 slot() const { return static_cast<uint32>(_packed); }
    [[nodiscard]] uint32 generation() const { return static_cast<uint32>(_packed >> 32U); }

    uint64 _packed;

    template <typename NodeData, typename EdgeData> friend class TreeArena;
};

using NodeIndex = ArenaIndex<detail::NodeIndexTag>;
using EdgeIndex = ArenaIndex<detail::EdgeIndexTag>;

enum class ArenaCapacityPolicy : int8 { StopSearch = 0, FailSession = 1 };

struct TreeArenaConfiguration {
    uint32 nodeCapacity;
    uint32 edgeCapacity;
    ArenaCapacityPolicy capacityPolicy;

    void validate() const {
        if (nodeCapacity == 0 || edgeCapacity == 0) {
            throw std::invalid_argument("tree arena capacities must be positive");
        }
    }
};

struct EdgeSpan {
    EdgeIndex first;
    uint32 count;
};

struct TreeArenaTelemetry {
    uint64 nodeCapacityExhaustions;
    uint64 edgeCapacityExhaustions;
};

template <typename NodeData, typename EdgeData> class TreeArena {
public:
    struct Node {
        NodeData data;
        std::optional<NodeIndex> parent;
        std::optional<EdgeSpan> children;
    };

    struct Edge {
        EdgeData data;
        NodeIndex owner;
        std::optional<NodeIndex> child;
    };

    explicit TreeArena(TreeArenaConfiguration configuration)
        : _configuration(configuration), _nodes(configuration.nodeCapacity),
          _edges(configuration.edgeCapacity) {
        configuration.validate();
    }

    TreeArena(const TreeArena &) = delete;
    TreeArena &operator=(const TreeArena &) = delete;
    TreeArena(TreeArena &&) = delete;
    TreeArena &operator=(TreeArena &&) = delete;

    [[nodiscard]] NodeIndex reset(NodeData rootData) {
        advanceGeneration();
        clearLiveValues();
        _usedNodes = 0;
        _usedEdges = 0;
        _root.reset();
        const std::optional<NodeIndex> root = createNode(std::move(rootData), std::nullopt);
        assert(root.has_value());
        _root = *root;
        return *root;
    }

    [[nodiscard]] std::optional<NodeIndex> createNode(NodeData data,
                                                      std::optional<NodeIndex> parent) {
        if (parent.has_value()) {
            static_cast<void>(node(*parent));
        }
        if (_usedNodes == _configuration.nodeCapacity) {
            ++_telemetry.nodeCapacityExhaustions;
            return capacityFailure<NodeIndex>("tree node arena exhausted");
        }
        const uint32 slot = _usedNodes++;
        _nodes[slot].emplace(Node{
            .data = std::move(data),
            .parent = parent,
            .children = std::nullopt,
        });
        return makeIndex<NodeIndex>(slot);
    }

    [[nodiscard]] std::optional<EdgeSpan> createEdges(NodeIndex owner,
                                                      std::span<const EdgeData> edgeData) {
        Node &ownerNode = node(owner);
        if (ownerNode.children.has_value()) {
            throw std::logic_error("tree node already has an edge span");
        }
        if (edgeData.empty()) {
            throw std::invalid_argument("tree edge span must not be empty");
        }
        if (edgeData.size() > static_cast<std::size_t>(_configuration.edgeCapacity - _usedEdges)) {
            ++_telemetry.edgeCapacityExhaustions;
            return capacityFailure<EdgeSpan>("tree edge arena exhausted");
        }
        const uint32 firstSlot = _usedEdges;
        for (const EdgeData &data : edgeData) {
            _edges[_usedEdges++].emplace(Edge{.data = data, .owner = owner, .child = std::nullopt});
        }
        const EdgeSpan span{
            .first = makeIndex<EdgeIndex>(firstSlot),
            .count = static_cast<uint32>(edgeData.size()),
        };
        ownerNode.children = span;
        return span;
    }

    void setChild(EdgeIndex edgeIndex, NodeIndex childIndex) {
        Edge &selectedEdge = edge(edgeIndex);
        Node &selectedChild = node(childIndex);
        if (selectedEdge.child.has_value()) {
            throw std::logic_error("tree edge already has a materialized child");
        }
        if (selectedChild.parent.has_value() && selectedChild.parent != selectedEdge.owner) {
            throw std::logic_error("tree child belongs to a different parent");
        }
        selectedEdge.child = childIndex;
        selectedChild.parent = selectedEdge.owner;
    }

    void reroot(NodeIndex retainedRoot) {
        Node &retainedNode = node(retainedRoot);
        retainedNode.parent.reset();
        _root = retainedRoot;
    }

    [[nodiscard]] NodeIndex rootIndex() const {
        if (!_root.has_value()) {
            throw std::logic_error("tree arena has no root");
        }
        return *_root;
    }

    [[nodiscard]] Node &node(NodeIndex index) {
        validateIndex(index, _usedNodes, "tree node index");
        std::optional<Node> &slot = _nodes[index.slot()];
        assert(slot.has_value());
        return *slot;
    }

    [[nodiscard]] const Node &node(NodeIndex index) const {
        return const_cast<TreeArena *>(this)->node(index);
    }

    [[nodiscard]] Edge &edge(EdgeIndex index) {
        validateIndex(index, _usedEdges, "tree edge index");
        std::optional<Edge> &slot = _edges[index.slot()];
        assert(slot.has_value());
        return *slot;
    }

    [[nodiscard]] const Edge &edge(EdgeIndex index) const {
        return const_cast<TreeArena *>(this)->edge(index);
    }

    [[nodiscard]] EdgeIndex edgeIndex(const EdgeSpan &span, uint32 offset) const {
        validateIndex(span.first, _usedEdges, "tree edge span");
        if (offset >= span.count || span.first.slot() + offset >= _usedEdges) {
            throw std::out_of_range("tree edge offset is outside its span");
        }
        return makeIndex<EdgeIndex>(span.first.slot() + offset);
    }

    [[nodiscard]] uint32 usedNodeCount() const { return _usedNodes; }
    [[nodiscard]] uint32 usedEdgeCount() const { return _usedEdges; }
    [[nodiscard]] uint32 generation() const { return _generation; }
    [[nodiscard]] const TreeArenaTelemetry &telemetry() const { return _telemetry; }

private:
    template <typename Index> [[nodiscard]] Index makeIndex(uint32 slot) const {
        return Index((static_cast<uint64>(_generation) << 32U) | static_cast<uint64>(slot));
    }

    template <typename Index>
    void validateIndex(Index index, uint32 usedCount, const char *description) const {
        if (index.generation() != _generation) {
            throw std::logic_error(std::string("stale ") + description);
        }
        if (index.slot() >= usedCount) {
            throw std::out_of_range(std::string(description) + " is outside the arena");
        }
    }

    template <typename Result>
    [[nodiscard]] std::optional<Result> capacityFailure(const char *description) const {
        if (_configuration.capacityPolicy == ArenaCapacityPolicy::FailSession) {
            throw std::overflow_error(description);
        }
        return std::nullopt;
    }

    void clearLiveValues() {
        for (uint32 slot = 0; slot < _usedEdges; ++slot) {
            _edges[slot].reset();
        }
        for (uint32 slot = 0; slot < _usedNodes; ++slot) {
            _nodes[slot].reset();
        }
    }

    void advanceGeneration() {
        if (_generation == std::numeric_limits<uint32>::max()) {
            throw std::overflow_error("tree arena generation exhausted");
        }
        ++_generation;
    }

    TreeArenaConfiguration _configuration;
    std::vector<std::optional<Node>> _nodes;
    std::vector<std::optional<Edge>> _edges;
    uint32 _usedNodes = 0;
    uint32 _usedEdges = 0;
    uint32 _generation = 0;
    std::optional<NodeIndex> _root;
    TreeArenaTelemetry _telemetry{
        .nodeCapacityExhaustions = 0,
        .edgeCapacityExhaustions = 0,
    };
};

template <typename Action> class SearchScratch {
public:
    SearchScratch(uint32 maximumDepth, uint32 maximumActions)
        : _maximumDepth(maximumDepth), _maximumActions(maximumActions) {
        if (maximumDepth == 0 || maximumActions == 0) {
            throw std::invalid_argument("search scratch capacities must be positive");
        }
        _path.reserve(maximumDepth);
        _actions.reserve(maximumActions);
        _priors.reserve(maximumActions);
    }

    void reset() {
        _path.clear();
        _actions.clear();
        _priors.clear();
    }

    void pushPath(NodeIndex nodeIndex) {
        if (_path.size() == _maximumDepth) {
            throw std::overflow_error("search path scratch exhausted");
        }
        _path.push_back(nodeIndex);
    }

    void pushAction(Action action, double prior) {
        if (_actions.size() == _maximumActions) {
            throw std::overflow_error("search action scratch exhausted");
        }
        _actions.push_back(action);
        _priors.push_back(prior);
    }

    [[nodiscard]] std::span<const NodeIndex> path() const { return _path; }
    [[nodiscard]] std::span<const Action> actions() const { return _actions; }
    [[nodiscard]] std::span<const double> priors() const { return _priors; }

private:
    std::size_t _maximumDepth;
    std::size_t _maximumActions;
    std::vector<NodeIndex> _path;
    std::vector<Action> _actions;
    std::vector<double> _priors;
};

} // namespace az::search
