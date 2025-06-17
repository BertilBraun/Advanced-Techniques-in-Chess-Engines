#include "NodePool.h"

#include "MCTSNode.hpp"

NodePool::NodePool() {
    // Make space for a few chunk pointers:
    m_chunks.reserve(4);
    addChunk();
}

void NodePool::deallocateNode(const NodeId id) {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    assert(isLive(id));
    std::optional<MCTSNode> *ptr = slotPointer(id);
    // Call the destructor of the MCTSNode at this slot:
    if (ptr->has_value())
        ptr->reset(); // This will call the destructor
    m_freeList.push_back(id);
}

size_t NodePool::capacity() const { return m_chunks.size() * CHUNK_SIZE; }

size_t NodePool::liveNodeCount() const {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    return m_nextFreshId - m_freeList.size();
}

void NodePool::clear() {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    m_chunks.clear();
    m_freeList.clear();
    m_nextFreshId = 0;
    addChunk(); // Always keep at least one chunk
}

bool NodePool::isLive(const NodeId id) const {
    std::lock_guard<std::mutex> lock(m_poolMutex);
    return id < m_nextFreshId && !isFreed(id);
}

void NodePool::purge(std::vector<NodeId> idsToKeep) {
    // sort idsToKeep, then add all IDs that are not in idsToKeep to m_freeList
    std::lock_guard<std::mutex> lock(m_poolMutex);
    std::sort(idsToKeep.begin(), idsToKeep.end());
    m_freeList.clear();
    m_freeList.reserve(m_nextFreshId - idsToKeep.size());
    int index = 0;
    for (NodeId id = 0; id < m_nextFreshId; ++id) {
        if (index < static_cast<int>(idsToKeep.size()) && idsToKeep[index] == id) {
            index++;
        } else {
            const auto slot = slotPointer(id);
            if (slot->has_value()) slot->reset();
            m_freeList.push_back(id);
        }
    }
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