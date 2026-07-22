#pragma once

#include "common.hpp"

#include "InferenceClientTypes.hpp"

#include <memory_resource>
#include <span>

using EvalNodeIndex = std::uint64_t;
using EvalVisitCount = std::pair<int, int>;
using EvalVisitCounts = std::vector<EvalVisitCount>;

inline constexpr EvalNodeIndex INVALID_EVAL_NODE_INDEX = std::numeric_limits<EvalNodeIndex>::max();

enum class EvalEdgeStorage { NewDelete, UnsynchronizedPool };

struct EvalSearchStatistics {
    std::uint32_t visits = 0;
    float result_sum = 0.0F;
    std::uint32_t virtual_loss = 0;
};

struct EvalSearchEdge {
    Move move;
    float policy;
    EvalSearchStatistics statistics;
    EvalNodeIndex child = INVALID_EVAL_NODE_INDEX;

    EvalSearchEdge(const Move move, const float policy) : move(move), policy(policy) {}
};

class EvalEdgeBlock {
public:
    EvalEdgeBlock(EvalEdgeStorage storage, std::pmr::memory_resource *resource)
        : m_storage(storage == EvalEdgeStorage::NewDelete ? Storage{StandardVector{}}
                                                          : Storage{PolymorphicVector{resource}}) {}

    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] std::size_t size() const {
        return m_storage.index() == 0 ? std::get<StandardVector>(m_storage).size()
                                      : std::get<PolymorphicVector>(m_storage).size();
    }
    void reserve(const std::size_t capacity) {
        if (m_storage.index() == 0) {
            std::get<StandardVector>(m_storage).reserve(capacity);
        } else {
            std::get<PolymorphicVector>(m_storage).reserve(capacity);
        }
    }
    void emplaceBack(const Move move, const float policy) {
        if (m_storage.index() == 0) {
            std::get<StandardVector>(m_storage).emplace_back(move, policy);
        } else {
            std::get<PolymorphicVector>(m_storage).emplace_back(move, policy);
        }
    }
    [[nodiscard]] EvalSearchEdge &operator[](const std::size_t index) { return edges()[index]; }
    [[nodiscard]] const EvalSearchEdge &operator[](const std::size_t index) const {
        return edges()[index];
    }
    [[nodiscard]] std::span<EvalSearchEdge> edges() {
        if (m_storage.index() == 0) {
            StandardVector &selected = std::get<StandardVector>(m_storage);
            return {selected.data(), selected.size()};
        }
        PolymorphicVector &selected = std::get<PolymorphicVector>(m_storage);
        return {selected.data(), selected.size()};
    }
    [[nodiscard]] std::span<const EvalSearchEdge> edges() const {
        return const_cast<EvalEdgeBlock *>(this)->edges();
    }

private:
    using StandardVector = std::vector<EvalSearchEdge>;
    using PolymorphicVector = std::pmr::vector<EvalSearchEdge>;
    using Storage = std::variant<StandardVector, PolymorphicVector>;
    Storage m_storage;
};

struct EvalSearchNode {
    Board board;
    EvalEdgeBlock children;
    EvalNodeIndex parent = INVALID_EVAL_NODE_INDEX;
    std::uint32_t parent_edge = std::numeric_limits<std::uint32_t>::max();
    std::optional<WdlPrediction> network_outcome;
    bool expanded = false;
    bool evaluating = false;

    EvalSearchNode(Board board, EvalNodeIndex parent, std::uint32_t parentEdge,
                   EvalEdgeStorage edgeStorage, std::pmr::memory_resource *edgeResource)
        : board(std::move(board)), children(edgeStorage, edgeResource), parent(parent),
          parent_edge(parentEdge) {}

    [[nodiscard]] bool isTerminal() const { return board.isGameOver(); }
};

class EvalSearchTree {
public:
    explicit EvalSearchTree(
        Board rootBoard, std::size_t initialCapacity = 1'024,
        EvalEdgeStorage edgeStorage = EvalEdgeStorage::NewDelete,
        std::size_t maximumCapacity = std::numeric_limits<std::uint32_t>::max());

    EvalSearchTree(const EvalSearchTree &) = delete;
    EvalSearchTree &operator=(const EvalSearchTree &) = delete;
    EvalSearchTree(EvalSearchTree &&) = delete;
    EvalSearchTree &operator=(EvalSearchTree &&) = delete;

    [[nodiscard]] EvalNodeIndex rootIndex() const { return m_rootIndex; }
    [[nodiscard]] const Board &rootBoard() const { return node(m_rootIndex).board; }
    [[nodiscard]] const EvalSearchStatistics &rootStatistics() const { return m_rootStatistics; }
    [[nodiscard]] std::size_t capacity() const { return m_slots.size(); }
    [[nodiscard]] std::size_t maximumCapacity() const { return m_maximumCapacity; }
    [[nodiscard]] std::size_t liveNodeCount() const { return m_liveNodeCount; }
    [[nodiscard]] std::size_t totalEdgeCount() const;
    [[nodiscard]] std::size_t maximumLiveNodeCount() const { return m_maximumLiveNodeCount; }

    [[nodiscard]] EvalSearchNode &node(EvalNodeIndex index);
    [[nodiscard]] const EvalSearchNode &node(EvalNodeIndex index) const;
    [[nodiscard]] EvalSearchEdge &edge(EvalNodeIndex parentIndex, std::uint32_t edgeIndex);
    [[nodiscard]] const EvalSearchEdge &edge(EvalNodeIndex parentIndex,
                                             std::uint32_t edgeIndex) const;

    void expand(EvalNodeIndex nodeIndex, const std::vector<MoveScore> &moves,
                std::optional<WdlPrediction> outcome = std::nullopt);
    [[nodiscard]] EvalNodeIndex materializeChild(EvalNodeIndex parentIndex,
                                                 std::uint32_t edgeIndex);
    [[nodiscard]] std::uint32_t bestChildEdge(EvalNodeIndex parentIndex,
                                              float explorationConstant) const;
    [[nodiscard]] std::uint32_t preferredRootEdge() const;
    [[nodiscard]] EvalNodeIndex selectLeaf(float explorationConstant);

    void reserveLeaf(EvalNodeIndex leafIndex);
    void cancelReservation(EvalNodeIndex leafIndex);
    void completeReservation(EvalNodeIndex leafIndex, float result);
    void addVirtualLoss(EvalNodeIndex leafIndex);
    void cancelVirtualLoss(EvalNodeIndex leafIndex);
    void backPropagate(EvalNodeIndex leafIndex, float result);
    void backPropagateAndRemoveVirtualLoss(EvalNodeIndex leafIndex, float result);

    [[nodiscard]] EvalNodeIndex reroot(std::uint32_t edgeIndex);
    [[nodiscard]] int maximumDepth() const;
    [[nodiscard]] EvalVisitCounts gatherVisitCounts() const;
    [[nodiscard]] std::size_t evaluatingNodeCount() const;
    [[nodiscard]] std::uint64_t totalVirtualLoss() const;

private:
    struct NodeSlot {
        std::uint32_t generation = 0;
        std::optional<EvalSearchNode> value;
    };

    // The resource precedes node storage so every child vector dies before its allocator.
    std::pmr::unsynchronized_pool_resource m_edgePool;
    std::pmr::memory_resource *m_edgeResource;
    EvalEdgeStorage m_edgeStorage;
    std::vector<NodeSlot> m_slots;
    std::vector<std::uint32_t> m_freeSlots;
    std::size_t m_liveNodeCount = 0;
    std::size_t m_maximumLiveNodeCount = 0;
    std::size_t m_maximumCapacity;
    EvalNodeIndex m_rootIndex = INVALID_EVAL_NODE_INDEX;
    EvalSearchStatistics m_rootStatistics;

    [[nodiscard]] static EvalNodeIndex makeIndex(std::uint32_t slot, std::uint32_t generation);
    [[nodiscard]] static std::uint32_t slotIndex(EvalNodeIndex index);
    [[nodiscard]] static std::uint32_t generation(EvalNodeIndex index);
    void grow();
    [[nodiscard]] EvalNodeIndex allocateNode(Board board, EvalNodeIndex parent,
                                             std::uint32_t parentEdge);
    void releaseNode(EvalNodeIndex index);
    void reclaimSubtree(EvalNodeIndex index);
    [[nodiscard]] EvalSearchStatistics &statistics(EvalNodeIndex index);
    [[nodiscard]] const EvalSearchStatistics &statistics(EvalNodeIndex index) const;
    [[nodiscard]] EvalNodeIndex selectAvailableLeaf(EvalNodeIndex nodeIndex,
                                                    float explorationConstant);
};
