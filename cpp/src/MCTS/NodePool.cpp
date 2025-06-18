#include "NodePool.h"

#include "MCTSNode.hpp"

NodePool::NodePool() {
    // Make space for a few chunk pointers:
    m_chunks.reserve(4);
    addChunk();
}

void NodePool::deallocateNode(const NodeId id) {
    std::lock_guard lock(m_poolMutex);
    assert(isLive(id) && "NodePool::deallocateNode: NodeId is not live");
    std::optional<MCTSNode> *ptr = slotPointer(id);
    // Call the destructor of the MCTSNode at this slot:
    assert(ptr->has_value() && "NodePool::deallocateNode: NodeId is not live");
    ptr->reset(); // This will call the destructor
    m_freeList.push_back(id);
}

size_t NodePool::capacity() const { return m_chunks.size() * CHUNK_SIZE; }

size_t NodePool::liveNodeCount() const {
    std::lock_guard lock(m_poolMutex);
    return m_nextFreshId - m_freeList.size();
}

void NodePool::clear() {
    std::cout << "NodePool::clear: clearing pool with " << m_chunks.size() << " chunks and "
              << m_freeList.size() << " free slots." << std::endl;
    std::lock_guard lock(m_poolMutex);
    for (const auto &chunk : m_chunks) {
        // Free the memory allocated for each chunk
        // - will call the destructor of each optional<MCTSNode> in the chunk
        delete chunk;
    }
    m_chunks.clear();
    m_freeList.clear();
    m_nextFreshId = 0;
    addChunk(); // Always keep at least one chunk
}

bool NodePool::isLive(const NodeId id) const {
    std::lock_guard lock(m_poolMutex);
    return id < m_nextFreshId && !isFreed(id);
}

void NodePool::purge(const std::vector<NodeId> &idsToKeep) {
    std::lock_guard lock(m_poolMutex);

    std::vector<bool> present(m_nextFreshId, false);
    for (const NodeId id : idsToKeep) {
        assert(id < m_nextFreshId && "NodeId out of range");
        present[id] = true;
    }

    m_freeList.clear();
    m_freeList.reserve(m_nextFreshId - idsToKeep.size());
    for (NodeId id = 0; id < m_nextFreshId; ++id) {
        if (!present[id]) {
            const auto slot = slotPointer(id);
            if (slot->has_value())
                slot->reset();
            m_freeList.push_back(id);
        }
    }
}

bool NodePool::isFreed(const NodeId id) const {
    assert(id < m_nextFreshId && "NodeId out of range");
    return contains(m_freeList, id);
}

void NodePool::addChunk() {
    m_chunks.push_back(new std::array<std::optional<MCTSNode>, CHUNK_SIZE>());
}

std::optional<MCTSNode> *NodePool::slotPointer(const NodeId id) const {
    const size_t chunkIdx = static_cast<size_t>(id) / CHUNK_SIZE;
    const size_t slotIdx = static_cast<size_t>(id) % CHUNK_SIZE;
    // m_chunks[chunk_idx] is a unique_ptr<std::array<std::optional<MCTSNode>,CHUNK_SIZE>>
    return &((*m_chunks[chunkIdx])[slotIdx]);
}