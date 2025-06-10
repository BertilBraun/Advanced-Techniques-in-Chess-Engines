#include "NodePool.h"

#include "MCTSNode.hpp"

NodePool::NodePool() {
    // Make space for a few chunk pointers:
    m_chunks.reserve(4);
    addChunk();
}

void NodePool::deallocateNode(const NodeId id) {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    assert(id < m_nextFreshId);
    std::optional<MCTSNode> *ptr = slotPointer(id);
    assert(ptr->has_value() && "Node is not allocated");
    // Call the destructor of the MCTSNode at this slot:
    ptr->reset(); // This will call the destructor
    m_freeList.push_back(id);
}

size_t NodePool::capacity() const { return m_chunks.size() * CHUNK_SIZE; }

size_t NodePool::liveNodeCount() const {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    return m_nextFreshId - m_freeList.size();
}

bool NodePool::isFreed(const NodeId id) const {
    assert(id < m_nextFreshId && "NodeId out of range");
    return contains(m_freeList, id);
}

void NodePool::addChunk() {
    m_chunks.push_back(std::make_unique<std::array<std::optional<MCTSNode>, CHUNK_SIZE>>());
}

std::optional<MCTSNode> *NodePool::slotPointer(const NodeId id) const {
    const size_t chunkIdx = static_cast<size_t>(id) / CHUNK_SIZE;
    const size_t slotIdx = static_cast<size_t>(id) % CHUNK_SIZE;
    // m_chunks[chunk_idx] is a unique_ptr<std::array<std::optional<MCTSNode>,CHUNK_SIZE>>
    return &((*m_chunks[chunkIdx])[slotIdx]);
}