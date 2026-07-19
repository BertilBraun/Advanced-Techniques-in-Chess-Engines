#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <typename KeyType, typename ValueType, std::size_t NumBuckets = 16,
          typename Hash = std::hash<KeyType>, typename KeyEqual = std::equal_to<KeyType>>
class ShardedCache {
public:
    explicit ShardedCache(const std::size_t maximumSize) : m_maximumSize(maximumSize) {
        if (maximumSize == 0) {
            throw std::invalid_argument("ShardedCache maximum size must be positive");
        }

        const std::size_t minimumBucketCapacity = maximumSize / NumBuckets;
        const std::size_t bucketsWithExtraEntry = maximumSize % NumBuckets;
        for (std::size_t index = 0; index < NumBuckets; ++index) {
            Bucket &bucket = m_buckets[index];
            bucket.capacity = minimumBucketCapacity + (index < bucketsWithExtraEntry ? 1 : 0);
            bucket.map.reserve(bucket.capacity);
        }
    }

    ShardedCache(const ShardedCache &) = delete;
    ShardedCache &operator=(const ShardedCache &) = delete;

    // Acquires a lease for key, inserting initialValue when key is absent.
    // Returns true only when the caller inserted the key and must produce its final value.
    [[nodiscard]] bool acquireOrInsert(const KeyType &key, const ValueType &initialValue) {
        Bucket &bucket = getBucket(key);
        std::lock_guard lock(bucket.mutex);
        const auto existing = bucket.map.find(key);
        if (existing != bucket.map.end()) {
            CacheEntry &entry = existing->second;
            if (entry.leaseCount == 0) {
                bucket.recency.erase(entry.recencyPosition);
            }
            ++entry.leaseCount;
            return false;
        }

        const auto [inserted, wasInserted] =
            bucket.map.try_emplace(key, CacheEntry{initialValue, 1});
        if (!wasInserted) {
            throw std::logic_error("ShardedCache insertion invariant failed");
        }
        return true;
    }

    // Updates a leased entry without making it eligible for eviction.
    void update(const KeyType &key, const ValueType &value) {
        Bucket &bucket = getBucket(key);
        std::lock_guard lock(bucket.mutex);
        const auto entry = bucket.map.find(key);
        if (entry == bucket.map.end() || entry->second.leaseCount == 0) {
            throw std::logic_error("ShardedCache update requires a leased entry");
        }
        entry->second.value = value;
    }

    // Reads a leased entry. The caller must release its lease after consuming the value.
    [[nodiscard]] bool lookup(const KeyType &key, ValueType &result) const {
        const Bucket &bucket = getBucket(key);
        std::shared_lock lock(bucket.mutex);
        const auto entry = bucket.map.find(key);
        if (entry == bucket.map.end()) {
            return false;
        }
        if (entry->second.leaseCount == 0) {
            throw std::logic_error("ShardedCache lookup requires a leased entry");
        }
        result = entry->second.value;
        return true;
    }

    // Releases a lease and promotes the entry to most-recently-used when no leases remain.
    void release(const KeyType &key) {
        Bucket &bucket = getBucket(key);
        std::lock_guard lock(bucket.mutex);
        const auto entry = bucket.map.find(key);
        if (entry == bucket.map.end() || entry->second.leaseCount == 0) {
            throw std::logic_error("ShardedCache release requires a leased entry");
        }

        --entry->second.leaseCount;
        if (entry->second.leaseCount != 0) {
            return;
        }

        bucket.recency.push_back(&entry->first);
        entry->second.recencyPosition = std::prev(bucket.recency.end());
        evictToCapacity(bucket);
    }

    void clear() {
        for (Bucket &bucket : m_buckets) {
            std::lock_guard lock(bucket.mutex);
            bucket.recency.clear();
            Map emptyMap;
            emptyMap.reserve(bucket.capacity);
            bucket.map.swap(emptyMap);
        }
    }

    [[nodiscard]] std::size_t size() const {
        std::size_t totalSize = 0;
        for (const Bucket &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex);
            totalSize += bucket.map.size();
        }
        return totalSize;
    }

    [[nodiscard]] bool empty() const { return size() == 0; }

    [[nodiscard]] std::size_t maximumSize() const { return m_maximumSize; }

    [[nodiscard]] std::size_t evictionCount() const {
        std::size_t totalEvictions = 0;
        for (const Bucket &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex);
            totalEvictions += bucket.evictions;
        }
        return totalEvictions;
    }

    [[nodiscard]] std::size_t estimatedStaticSizeBytes() const {
        std::size_t totalBytes = 0;
        for (const Bucket &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex);
            totalBytes += bucket.map.bucket_count() * sizeof(void *);
            totalBytes +=
                bucket.map.size() * (sizeof(typename Map::value_type) + 2 * sizeof(void *));
            totalBytes += bucket.recency.size() * 3 * sizeof(void *);
        }
        return totalBytes;
    }

    template <typename Visitor> void forEachValue(Visitor visitor) const {
        for (const Bucket &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex);
            for (const auto &[key, entry] : bucket.map) {
                static_cast<void>(key);
                visitor(entry.value);
            }
        }
    }

private:
    using RecencyList = std::list<const KeyType *>;

    struct CacheEntry {
        ValueType value;
        std::size_t leaseCount;
        typename RecencyList::iterator recencyPosition{};
    };

    using Map = std::unordered_map<KeyType, CacheEntry, Hash, KeyEqual>;

    struct Bucket {
        Map map;
        RecencyList recency;
        mutable std::shared_mutex mutex;
        std::size_t capacity = 0;
        std::size_t evictions = 0;
    };

    std::array<Bucket, NumBuckets> m_buckets;
    std::size_t m_maximumSize;
    Hash m_hasher{};

    [[nodiscard]] Bucket &getBucket(const KeyType &key) {
        return m_buckets[m_hasher(key) % NumBuckets];
    }

    [[nodiscard]] const Bucket &getBucket(const KeyType &key) const {
        return m_buckets[m_hasher(key) % NumBuckets];
    }

    static void evictToCapacity(Bucket &bucket) {
        while (bucket.map.size() > bucket.capacity && !bucket.recency.empty()) {
            const KeyType *leastRecentlyUsedKey = bucket.recency.front();
            const auto entry = bucket.map.find(*leastRecentlyUsedKey);
            if (entry == bucket.map.end()) {
                throw std::logic_error("ShardedCache recency invariant failed");
            }
            bucket.recency.pop_front();
            bucket.map.erase(entry);
            ++bucket.evictions;
        }
    }
};
