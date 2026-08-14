#pragma once

#include "games/GameConcepts.hpp"
#include "search/tree/TreeArena.hpp"
#include "util/py.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace search_tree_detail {

[[nodiscard]] inline std::vector<std::uint32_t>
allocateRetainedVisits(const std::vector<std::uint32_t> &bucketVisits,
                       const std::uint32_t retainedVisits) {
    const std::uint64_t totalVisits =
        std::accumulate(bucketVisits.begin(), bucketVisits.end(), std::uint64_t{0});
    if (retainedVisits > totalVisits) {
        throw std::logic_error("Retained game-search visits exceed original visits");
    }
    std::vector<std::uint32_t> result(bucketVisits.size(), 0);
    if (totalVisits == 0) {
        return result;
    }
    struct Remainder {
        std::uint64_t amount;
        std::uint32_t original_visits;
        std::size_t index;
    };
    std::vector<Remainder> remainders;
    remainders.reserve(bucketVisits.size());
    std::uint64_t allocated = 0;
    for (const auto index : range(bucketVisits.size())) {
        const std::uint64_t numerator =
            static_cast<std::uint64_t>(bucketVisits[index]) * retainedVisits;
        result[index] = static_cast<std::uint32_t>(numerator / totalVisits);
        allocated += result[index];
        remainders.push_back({
            .amount = numerator % totalVisits,
            .original_visits = bucketVisits[index],
            .index = index,
        });
    }
    std::ranges::sort(remainders, [](const Remainder &left, const Remainder &right) {
        if (left.amount != right.amount) {
            return left.amount > right.amount;
        }
        if (left.original_visits != right.original_visits) {
            return left.original_visits > right.original_visits;
        }
        return left.index < right.index;
    });
    for (std::uint64_t offset = 0; offset < retainedVisits - allocated; ++offset) {
        ++result[remainders[offset].index];
    }
    return result;
}

[[nodiscard]] inline float retainValueSum(const float originalValueSum,
                                          const std::uint32_t originalVisits,
                                          const std::uint32_t retainedVisits) {
    if (originalVisits == 0 || retainedVisits == 0) {
        return 0.0F;
    }
    const float retained =
        originalValueSum * static_cast<float>(retainedVisits) / static_cast<float>(originalVisits);
    return std::clamp(retained, -static_cast<float>(retainedVisits),
                      static_cast<float>(retainedVisits));
}

template <SearchGame Game>
void retainSubtree(TreeArena<Game> &arena, const std::size_t nodeIndex,
                   const std::uint32_t retainedVisits, const float valueDiscountPerPly) {
    using Edge = GameSearchEdge<typename Game::Action>;
    GameSearchNode<Game> &selected = arena.node(nodeIndex);
    const std::uint32_t originalVisits = selected.visits;
    std::uint64_t childVisits = 0;
    float childValueSum = 0.0F;
    for (const Edge &edge : selected.children) {
        childVisits += edge.visits;
        childValueSum += edge.value_sum;
    }
    if (childVisits > originalVisits) {
        throw std::logic_error("Game-search child visits exceed parent visits");
    }
    const std::uint32_t localVisits = originalVisits - static_cast<std::uint32_t>(childVisits);
    const float localValueSum = selected.value_sum + valueDiscountPerPly * childValueSum;
    std::vector<std::uint32_t> buckets;
    buckets.reserve(selected.children.size() + 1);
    buckets.push_back(localVisits);
    for (const Edge &edge : selected.children) {
        buckets.push_back(edge.visits);
    }
    const std::vector<std::uint32_t> retainedBuckets =
        allocateRetainedVisits(buckets, retainedVisits);
    const float retainedLocalValue =
        retainValueSum(localValueSum, localVisits, retainedBuckets.front());

    float retainedChildValueSum = 0.0F;
    for (const auto edgeIndex : range(selected.children.size())) {
        Edge &edge = arena.node(nodeIndex).children[edgeIndex];
        const std::uint32_t originalEdgeVisits = edge.visits;
        const float originalEdgeValue = edge.value_sum;
        const std::uint32_t retainedEdgeVisits = retainedBuckets[edgeIndex + 1];
        if (edge.child_index.has_value()) {
            const std::size_t childIndex = *edge.child_index;
            const GameSearchNode<Game> &child = arena.node(childIndex);
            if (child.visits != originalEdgeVisits ||
                std::abs(child.value_sum - originalEdgeValue) > 1e-4F) {
                throw std::logic_error(
                    "Materialized game-search child disagrees with its incoming edge");
            }
            if (retainedEdgeVisits == 0) {
                arena.reclaimSubtree(childIndex);
                arena.node(nodeIndex).children[edgeIndex].child_index.reset();
                edge.visits = 0;
                edge.value_sum = 0.0F;
            } else {
                retainSubtree(arena, childIndex, retainedEdgeVisits, valueDiscountPerPly);
                edge.visits = arena.node(childIndex).visits;
                edge.value_sum = arena.node(childIndex).value_sum;
            }
        } else {
            edge.visits = retainedEdgeVisits;
            edge.value_sum =
                retainValueSum(originalEdgeValue, originalEdgeVisits, retainedEdgeVisits);
        }
        retainedChildValueSum += edge.value_sum;
    }
    GameSearchNode<Game> &retained = arena.node(nodeIndex);
    retained.visits = retainedVisits;
    retained.value_sum = retainedLocalValue - valueDiscountPerPly * retainedChildValueSum;
}

} // namespace search_tree_detail
