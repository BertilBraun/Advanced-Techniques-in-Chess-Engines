#pragma once

#include "common.hpp"

#include "MCTSNode.hpp"

static constexpr size_t CHUNK_SIZE = 1024; // Default size of each chunk in the pool

///
/// NodePool: “chunked arena” with free‐list + thread safety,
/// where each chunk is a heap‐allocated std::array<std::optional<MCTSNode>,CHUNK_SIZE>.
///
class NodePool {
public:
    NodePool();

    ~NodePool() {
        clear();
    }

    // Allocate a new node by forwarding to MCTSNode’s constructor. Returns its NodeId.
    // Thread‐safe: locks the internal mutex while modifying m_chunks/m_freeList/m_nextFreshId.
    template <typename... Args> [[nodiscard]] MCTSNode *allocateNode(Args &&...args);

    // Deallocate a node: call its destructor, then push its ID to m_freeList.
    // Thread‐safe.
    void deallocateNode(NodeId id);

    // Get a mutable reference to a MCTSNode (not thread‐safe by itself;
    // if you mutate fields, you should lock node_mutex inside MCTSNode).
    [[nodiscard]] MCTSNode *get(const NodeId id) {
        TIMEIT("NodePool::get");
        assert(isLive(id) && slotPointer(id)->has_value() && "NodeId is not live");
        return &slotPointer(id)->value();
    }

    [[nodiscard]] const MCTSNode *get(const NodeId id) const {
        TIMEIT("NodePool::get const");
        assert(isLive(id) && slotPointer(id)->has_value() && "NodeId is not live");
        return &slotPointer(id)->value();
    }

    // How many NodeIds have ever been “touched” (including freed)?
    [[nodiscard]] size_t capacity() const;

    // How many nodes are currently live (allocated but not freed)?
    [[nodiscard]] size_t liveNodeCount() const;

    void clear();

    [[nodiscard]] bool isLive(NodeId id) const;

    void purge(const std::vector<NodeId> &idsToKeep);

private:
    // Each entry is a unique_ptr to a heap‐allocated
    // std::array<std::optional<MCTSNode>,CHUNK_SIZE>.
    std::vector<std::array<std::optional<MCTSNode>, CHUNK_SIZE>*> m_chunks;

    // IDs that have been freed and can be reused.
    std::vector<NodeId> m_freeList;

    // Next fresh ID (0...m_nextFreshId-1 have all been touched at least once).
    NodeId m_nextFreshId = 0;

    // Guards any change to m_chunks, m_freeList, m_nextFreshId.
    mutable std::mutex m_poolMutex;

    // Return true if id is in m_freeList (i.e. not currently live):
    [[nodiscard]] bool isFreed(NodeId id) const;

    // Add one more chunk: allocate a new std::array<std::optional<MCTSNode>,CHUNK_SIZE> on the
    // heap.
    void addChunk();

    // Placement‐construct a MCTSNode at slot `id`.
    template <typename... Args> [[nodiscard]] MCTSNode* constructAt(NodeId id, Args &&...args);

    // Given a NodeId, return the address of its slot in the correct chunk:
    [[nodiscard]] std::optional<MCTSNode> *slotPointer(NodeId id) const;
};

template <typename... Args> MCTSNode *NodePool::allocateNode(Args &&...args) {
    NodeId newId;
    {
        std::lock_guard lock(m_poolMutex);

        if (!m_freeList.empty()) {
            newId = m_freeList.back();
            m_freeList.pop_back();
        } else {
            // No free slot: get the next fresh ID
            newId = m_nextFreshId++;
            const size_t chunkIdx = static_cast<size_t>(newId) / CHUNK_SIZE;
            if (chunkIdx >= m_chunks.size()) {
                addChunk();
            }
        }
    }
    return constructAt(newId, std::forward<Args>(args)...);
}

template <typename... Args> MCTSNode* NodePool::constructAt(const NodeId id, Args &&...args) {
    assert(!isLive(id) && "Node already allocated at this ID");
    std::optional<MCTSNode> *opt = slotPointer(id);
    assert(!opt->has_value() && "Node already allocated at this ID");
    opt->emplace(std::forward<Args>(args)...);
    opt->value().myId = id;
    return &opt->value();
}
