#include "NodePool.h"

#include "MCTSNode.hpp"

NodePool::NodePool() {
    // Make space for a few chunk pointers:
    chunks_.reserve(4);
    addChunk();
}

NodePool::~NodePool() {
    // We need to explicitly destroy any live MCTSNode (i.e. placement‐new’d ones),
    // then let unique_ptr<> automatically free each std::array.

    // For every NodeId < next_fresh_id_, if it’s not in free_list_, call its destructor:
    for (NodeId id = 0; id < next_fresh_id_; ++id) {
        if (!isFreed(id)) {
            MCTSNode *ptr = slotPointer(id);
            ptr->~MCTSNode();
        }
    }
    chunks_.clear();
}

void NodePool::deallocateNode(NodeId id) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    assert(id < next_fresh_id_);
    MCTSNode *ptr = slotPointer(id);
    ptr->~MCTSNode();
    free_list_.push_back(id);
}

MCTSNode &NodePool::get(NodeId id) {
    assert(id < next_fresh_id_);
    return *slotPointer(id);
}

const MCTSNode &NodePool::get(NodeId id) const {
    assert(id < next_fresh_id_);
    return *slotPointer(id);
}

size_t NodePool::capacity() const { return chunks_.size() * CHUNK_SIZE; }

size_t NodePool::liveNodeCount() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return next_fresh_id_ - free_list_.size();
}

bool NodePool::isFreed(NodeId id) const {
    assert(id < next_fresh_id_, "NodeId out of range");
    return contains(free_list_, id);
}

void NodePool::addChunk() {
    chunks_.push_back(std::make_unique<std::array<MCTSNode, CHUNK_SIZE>>());
}

MCTSNode *NodePool::slotPointer(NodeId id) const {
    const size_t chunk_idx = static_cast<size_t>(id) / CHUNK_SIZE;
    const size_t slot_idx = static_cast<size_t>(id) % CHUNK_SIZE;
    // chunks_[chunk_idx] is a unique_ptr<std::array<MCTSNode,CHUNK_SIZE>>
    return &((*chunks_[chunk_idx])[slot_idx]);
}