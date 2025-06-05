#pragma once

#include "common.hpp"

#include "MCTSNode.hpp"

static constexpr size_t CHUNK_SIZE = 1024; // Default size of each chunk in the pool

///
/// NodePool: “chunked arena” with free‐list + thread safety,
/// where each chunk is a heap‐allocated std::array<MCTSNode,CHUNK_SIZE>.
///
class NodePool {
public:
    NodePool();

    ~NodePool();

    // Allocate a new node by forwarding to MCTSNode’s constructor. Returns its NodeId.
    // Thread‐safe: locks the internal mutex while modifying chunks_/free_list_/next_fresh_id_.
    template <typename... Args> NodeId allocateNode(Args &&...args);

    // Deallocate a node: call its destructor, then push its ID to free_list_.
    // Thread‐safe.
    void deallocateNode(NodeId id);

    // Get a mutable reference to a MCTSNode (not thread‐safe by itself;
    // if you mutate fields, you should lock node_mutex inside MCTSNode).
    MCTSNode &get(NodeId id);

    const MCTSNode &get(NodeId id) const;

    // How many NodeIds have ever been “touched” (including freed)?
    size_t capacity() const;

    // How many nodes are currently live (allocated but not freed)?
    size_t liveNodeCount() const;

private:
    // Each entry is a unique_ptr to a heap‐allocated std::array<MCTSNode,CHUNK_SIZE>.
    std::vector<std::unique_ptr<std::array<MCTSNode, CHUNK_SIZE>>> chunks_;

    // IDs that have been freed and can be reused.
    std::vector<NodeId> free_list_;

    // Next fresh ID (0.. next_fresh_id_-1 have all been touched at least once).
    NodeId next_fresh_id_ = 0;

    // Guards any change to chunks_, free_list_, next_fresh_id_.
    mutable std::mutex pool_mutex_;

    // Return true if id is in free_list_ (i.e. not currently live):
    bool isFreed(NodeId id) const;

    // Add one more chunk: allocate a new std::array<MCTSNode,CHUNK_SIZE> on the heap.
    void addChunk();

    // Placement‐construct a MCTSNode at slot `id`.
    template <typename... Args> void constructAt(NodeId id, Args &&...args);

    // Given a NodeId, return the address of its slot in the correct chunk:
    MCTSNode *slotPointer(NodeId id) const;
};

template <typename... Args> NodeId NodePool::allocateNode(Args &&...args) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!free_list_.empty()) {
        NodeId reuse_id = free_list_.back();
        free_list_.pop_back();
        constructAt(reuse_id, std::forward<Args>(args)...);
        return reuse_id;
    }

    // No free slot: get the next fresh ID
    const NodeId id = next_fresh_id_;
    const size_t chunk_idx = static_cast<size_t>(id) / CHUNK_SIZE;
    if (chunk_idx >= chunks_.size()) {
        addChunk();
    }
    ++next_fresh_id_;
    constructAt(id, std::forward<Args>(args)...);
    return id;
}

template <typename... Args> void NodePool::constructAt(NodeId id, Args &&...args) {
    MCTSNode *ptr = slotPointer(id);
    new (ptr) MCTSNode(std::forward<Args>(args)...);
}
