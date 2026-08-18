#pragma once

#include "search/InferenceDimensions.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_set>

// A 128-bit hash-only upper bound over exact encoded inputs; collisions are negligibly likely.

struct InferenceCacheStatistics {
    std::uint64_t total_positions = 0;
    std::uint64_t unique_hashes = 0;
    std::uint64_t repeated_hashes = 0;
    std::uint64_t same_batch_repeats = 0;
    std::uint64_t prior_batch_repeats = 0;
    std::uint64_t set_size = 0;
    double repeat_rate = 0.0;
};

class InferenceCacheTracker {
public:
    explicit InferenceCacheTracker(InferenceDimensions dimensions);

    void observeBatch(const std::int8_t *encodedInputs, std::size_t batchSize);
    void reset();
    [[nodiscard]] InferenceCacheStatistics statistics() const;

private:
    struct InputHash {
        std::uint64_t low;
        std::uint64_t high;

        [[nodiscard]] bool operator==(const InputHash &) const noexcept = default;
    };

    struct InputHashHasher {
        [[nodiscard]] std::size_t operator()(const InputHash &hash) const noexcept;
    };

    [[nodiscard]] InputHash hashInput(const std::int8_t *encodedInput) const noexcept;

    const InferenceDimensions m_dimensions;
    mutable std::mutex m_mutex;
    std::unordered_set<InputHash, InputHashHasher> m_hashes;
    std::unordered_set<InputHash, InputHashHasher> m_batchHashes;
    std::uint64_t m_totalPositions = 0;
    std::uint64_t m_repeatedHashes = 0;
    std::uint64_t m_sameBatchRepeats = 0;
    std::uint64_t m_priorBatchRepeats = 0;
};
