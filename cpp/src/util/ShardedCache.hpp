#pragma once

#include "common.hpp"
/* ────────────────────────────────────────────────────────────────
 *  template parameters
 *    KeyType      – key
 *    ValueType    – mapped value
 *    NumBuckets   – number of shards (compile-time constant)
 *    Hash         – hasher   (defaults to std::hash<KeyType>)
 *    KeyEqual     – equality (defaults to std::equal_to<KeyType>)
 * ──────────────────────────────────────────────────────────────── */
template <typename KeyType, typename ValueType, std::size_t NumBuckets = 16,
          typename Hash = std::hash<KeyType>, typename KeyEqual = std::equal_to<KeyType>>
class ShardedCache {
public:
    ShardedCache() {
        // Initialize buckets.
        for (auto &bucket : m_buckets)
            bucket.map.reserve(1024); // Preallocate space for 1024 elements.
    }

    // Look up a value. Returns true and sets result if found.
    bool lookup(const KeyType &key, ValueType &result) {
        auto &bucket = getBucket(key);
        std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
        auto it = bucket.map.find(key);
        if (it != bucket.map.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    [[nodiscard]] bool contains(const KeyType &key) {
        auto &bucket = getBucket(key);
        std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
        return bucket.map.find(key) != bucket.map.end();
    }

    // Insert a new key/value pair.
    void insert(const KeyType &key, const ValueType &result) {
        auto &bucket = getBucket(key);
        std::lock_guard lock(bucket.mutex); // Exclusive lock for writing.
        bucket.map[key] = result;
    }

    // Insert a new key/value pair only if the key is not already present.
    void insertIfNotPresent(const KeyType &key, const ValueType &result) {
        auto &bucket = getBucket(key);
        std::lock_guard lock(bucket.mutex); // Exclusive lock for writing.
        if (bucket.map.find(key) == bucket.map.end()) {
            bucket.map[key] = result;
        }
    }

    void clear() {
        for (auto &bucket : m_buckets) {
            std::lock_guard lock(bucket.mutex); // Exclusive lock for writing.
            bucket.map.clear();
        }
    }

    [[nodiscard]] size_t size() const {
        size_t totalSize = 0;
        for (const auto &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
            totalSize += bucket.map.size();
        }
        return totalSize;
    }

    // Check if the cache is empty.
    [[nodiscard]] bool empty() const {
        for (const auto &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
            if (!bucket.map.empty()) {
                return false;
            }
        }
        return true;
    }

    // Iterator that locks all buckets during iteration
    class LockedIterator {
    private:
        ShardedCache &m_cache;
        std::vector<std::shared_lock<std::shared_mutex>> m_locks;
        using MapIterator =
            typename std::unordered_map<KeyType, ValueType, Hash, KeyEqual>::iterator;
        size_t m_currentBucket = 0;
        MapIterator m_currentIt;
        MapIterator m_currentEnd;

    public:
        using value_type = std::pair<const KeyType, ValueType>;
        using reference = value_type &;
        using pointer = value_type *;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        explicit LockedIterator(ShardedCache &c, bool isEnd = false) : m_cache(c) {
            // Lock all buckets
            for (size_t i = 0; i < NumBuckets; ++i) {
                m_locks.emplace_back(m_cache.m_buckets[i].mutex);
            }

            if (!isEnd) {
                // Find first non-empty bucket
                while (m_currentBucket < NumBuckets &&
                       m_cache.m_buckets[m_currentBucket].map.empty()) {
                    ++m_currentBucket;
                }

                if (m_currentBucket < NumBuckets) {
                    m_currentIt = m_cache.m_buckets[m_currentBucket].map.begin();
                    m_currentEnd = m_cache.m_buckets[m_currentBucket].map.end();
                }
            } else {
                m_currentBucket = NumBuckets;
            }
        }

        // Move to next element, possibly crossing bucket boundaries
        LockedIterator &operator++() {
            if (m_currentBucket < NumBuckets) {
                ++m_currentIt;
                if (m_currentIt == m_currentEnd) {
                    // Find next non-empty bucket
                    do {
                        ++m_currentBucket;
                    } while (m_currentBucket < NumBuckets &&
                             m_cache.m_buckets[m_currentBucket].map.empty());

                    if (m_currentBucket < NumBuckets) {
                        m_currentIt = m_cache.m_buckets[m_currentBucket].map.begin();
                        m_currentEnd = m_cache.m_buckets[m_currentBucket].map.end();
                    }
                }
            }
            return *this;
        }

        reference operator*() { return *m_currentIt; }
        pointer operator->() { return &(*m_currentIt); }

        bool operator==(const LockedIterator &other) const {
            if (m_currentBucket != other.m_currentBucket)
                return false;
            return (m_currentBucket == NumBuckets) || (m_currentIt == other.m_currentIt);
        }

        bool operator!=(const LockedIterator &other) const { return !(*this == other); }

        // Locks are automatically released when iterator is destroyed
    };

    [[nodiscard]] LockedIterator begin() { return LockedIterator(*this); }
    [[nodiscard]] LockedIterator end() { return LockedIterator(*this, true); }

private:
    struct Bucket {
        std::unordered_map<KeyType, ValueType, Hash, KeyEqual> map;
        mutable std::shared_mutex mutex;
    };

    std::array<Bucket, NumBuckets> m_buckets;
    Hash m_hasher{}; // one copy – avoids constructing every call

    [[nodiscard]] Bucket &getBucket(const KeyType &key) {
        // Simple hash to bucket index.
        return m_buckets[m_hasher(key) % NumBuckets];
    }
};
