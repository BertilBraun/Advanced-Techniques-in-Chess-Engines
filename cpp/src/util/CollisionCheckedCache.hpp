#pragma once

#include "ShardedCache.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

enum class CacheAcquisition {
    Inserted,
    Hit,
    FingerprintCollision,
};

template <typename FingerprintType, typename IdentityType, typename ValueType,
          std::size_t NumBuckets = 16, typename Hash = std::hash<FingerprintType>>
class CollisionCheckedCache {
public:
    explicit CollisionCheckedCache(const std::size_t maximumSize) : m_cache(maximumSize) {}

    [[nodiscard]] CacheAcquisition acquireOrInsert(const FingerprintType &fingerprint,
                                                   const IdentityType &identity,
                                                   const ValueType &initialValue) {
        const Entry initialEntry{identity, initialValue};
        if (m_cache.acquireOrInsert(fingerprint, initialEntry)) {
            return CacheAcquisition::Inserted;
        }

        Entry cachedEntry;
        if (!m_cache.lookup(fingerprint, cachedEntry)) {
            throw std::logic_error("CollisionCheckedCache lost a leased entry");
        }
        if (cachedEntry.identity == identity) {
            return CacheAcquisition::Hit;
        }

        m_cache.release(fingerprint);
        return CacheAcquisition::FingerprintCollision;
    }

    void update(const FingerprintType &fingerprint, const IdentityType &identity,
                const ValueType &value) {
        Entry cachedEntry;
        if (!m_cache.lookup(fingerprint, cachedEntry) || !(cachedEntry.identity == identity)) {
            throw std::logic_error("CollisionCheckedCache update identity mismatch");
        }
        m_cache.update(fingerprint, Entry{identity, value});
    }

    [[nodiscard]] bool lookup(const FingerprintType &fingerprint, const IdentityType &identity,
                              ValueType &result) const {
        Entry cachedEntry;
        if (!m_cache.lookup(fingerprint, cachedEntry)) {
            return false;
        }
        if (!(cachedEntry.identity == identity)) {
            throw std::logic_error("CollisionCheckedCache lookup identity mismatch");
        }
        result = std::move(cachedEntry.value);
        return true;
    }

    void release(const FingerprintType &fingerprint) { m_cache.release(fingerprint); }

    [[nodiscard]] std::size_t size() const { return m_cache.size(); }
    [[nodiscard]] bool empty() const { return m_cache.empty(); }
    [[nodiscard]] std::size_t maximumSize() const { return m_cache.maximumSize(); }
    [[nodiscard]] std::size_t evictionCount() const { return m_cache.evictionCount(); }
    [[nodiscard]] std::size_t estimatedStaticSizeBytes() const {
        return m_cache.estimatedStaticSizeBytes();
    }

    template <typename Visitor> void forEachValue(Visitor visitor) const {
        m_cache.forEachValue(
            [&visitor](const Entry &entry) { visitor(entry.identity, entry.value); });
    }

private:
    struct Entry {
        IdentityType identity;
        ValueType value;
    };

    ShardedCache<FingerprintType, Entry, NumBuckets, Hash> m_cache;
};
