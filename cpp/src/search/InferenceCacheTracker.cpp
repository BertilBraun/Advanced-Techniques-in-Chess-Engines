#include "search/InferenceCacheTracker.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace {
constexpr std::uint64_t rotateLeft(const std::uint64_t value, const int bits) noexcept {
    return (value << bits) | (value >> (64 - bits));
}

constexpr std::uint64_t avalanche(std::uint64_t value) noexcept {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    return value ^ (value >> 33);
}

std::uint64_t loadWord(const std::uint8_t *bytes) noexcept {
    std::uint64_t word;
    std::memcpy(&word, bytes, sizeof(word));
    return word;
}

std::array<std::uint64_t, 2> hashBytes(const std::uint8_t *bytes, const std::size_t size,
                                       const std::uint64_t seed) noexcept {
    constexpr std::uint64_t firstMultiplier = 0x87c37b91114253d5ULL;
    constexpr std::uint64_t secondMultiplier = 0x4cf5ad432745937fULL;
    std::uint64_t first = seed;
    std::uint64_t second = seed;
    const std::size_t blockCount = size / 16;
    for (std::size_t block = 0; block < blockCount; ++block) {
        std::uint64_t firstWord = loadWord(bytes + block * 16);
        std::uint64_t secondWord = loadWord(bytes + block * 16 + 8);
        firstWord *= firstMultiplier;
        firstWord = rotateLeft(firstWord, 31);
        firstWord *= secondMultiplier;
        first ^= firstWord;
        first = rotateLeft(first, 27);
        first += second;
        first = first * 5 + 0x52dce729;
        secondWord *= secondMultiplier;
        secondWord = rotateLeft(secondWord, 33);
        secondWord *= firstMultiplier;
        second ^= secondWord;
        second = rotateLeft(second, 31);
        second += first;
        second = second * 5 + 0x38495ab5;
    }

    const std::uint8_t *tail = bytes + blockCount * 16;
    std::uint64_t firstTail = 0;
    std::uint64_t secondTail = 0;
    const std::size_t tailSize = size & 15;
    for (std::size_t index = 0; index < std::min<std::size_t>(tailSize, 8); ++index) {
        firstTail |= static_cast<std::uint64_t>(tail[index]) << (index * 8);
    }
    for (std::size_t index = 8; index < tailSize; ++index) {
        secondTail |= static_cast<std::uint64_t>(tail[index]) << ((index - 8) * 8);
    }
    if (firstTail != 0) {
        firstTail *= firstMultiplier;
        firstTail = rotateLeft(firstTail, 31);
        firstTail *= secondMultiplier;
        first ^= firstTail;
    }
    if (secondTail != 0) {
        secondTail *= secondMultiplier;
        secondTail = rotateLeft(secondTail, 33);
        secondTail *= firstMultiplier;
        second ^= secondTail;
    }
    first ^= size;
    second ^= size;
    first += second;
    second += first;
    first = avalanche(first);
    second = avalanche(second);
    first += second;
    second += first;
    return {first, second};
}

std::uint64_t dimensionsSeed(const InferenceDimensions dimensions) noexcept {
    std::uint64_t seed = 0x9368e53c2f6af274ULL;
    for (const std::size_t dimension : {dimensions.channels, dimensions.rows, dimensions.columns,
                                        dimensions.actions, dimensions.outcomes}) {
        seed = avalanche(seed ^ static_cast<std::uint64_t>(dimension));
    }
    return seed;
}
} // namespace

InferenceCacheTracker::InferenceCacheTracker(const InferenceDimensions dimensions)
    : m_dimensions(dimensions) {
    if (dimensions.channels == 0 || dimensions.rows == 0 || dimensions.columns == 0 ||
        dimensions.actions == 0 || dimensions.outcomes == 0) {
        throw std::invalid_argument("Inference cache dimensions must be positive");
    }
}

std::size_t
InferenceCacheTracker::InputHashHasher::operator()(const InputHash &hash) const noexcept {
    return static_cast<std::size_t>(avalanche(hash.low ^ rotateLeft(hash.high, 29)));
}

InferenceCacheTracker::InputHash
InferenceCacheTracker::hashInput(const std::int8_t *encodedInput) const noexcept {
    const auto hash = hashBytes(reinterpret_cast<const std::uint8_t *>(encodedInput),
                                m_dimensions.encodedSize(), dimensionsSeed(m_dimensions));
    return {.low = hash[0], .high = hash[1]};
}

void InferenceCacheTracker::observeBatch(const std::int8_t *encodedInputs,
                                         const std::size_t batchSize) {
    std::scoped_lock lock(m_mutex);
    m_batchHashes.clear();
    m_batchHashes.reserve(batchSize);
    for (std::size_t row = 0; row < batchSize; ++row) {
        const InputHash hash = hashInput(encodedInputs + row * m_dimensions.encodedSize());
        const bool firstInBatch = m_batchHashes.insert(hash).second;
        const bool firstOverall = m_hashes.insert(hash).second;
        ++m_totalPositions;
        if (firstOverall) {
            continue;
        }
        ++m_repeatedHashes;
        if (firstInBatch) {
            ++m_priorBatchRepeats;
        } else {
            ++m_sameBatchRepeats;
        }
    }
}

void InferenceCacheTracker::reset() {
    std::scoped_lock lock(m_mutex);
    m_hashes.clear();
    m_batchHashes.clear();
    m_totalPositions = 0;
    m_repeatedHashes = 0;
    m_sameBatchRepeats = 0;
    m_priorBatchRepeats = 0;
}

InferenceCacheStatistics InferenceCacheTracker::statistics() const {
    std::scoped_lock lock(m_mutex);
    const std::uint64_t uniqueHashes = m_hashes.size();
    return {
        .total_positions = m_totalPositions,
        .unique_hashes = uniqueHashes,
        .repeated_hashes = m_repeatedHashes,
        .same_batch_repeats = m_sameBatchRepeats,
        .prior_batch_repeats = m_priorBatchRepeats,
        .set_size = uniqueHashes,
        .repeat_rate = m_totalPositions == 0 ? 0.0
                                             : static_cast<double>(m_repeatedHashes) /
                                                   static_cast<double>(m_totalPositions),
    };
}
