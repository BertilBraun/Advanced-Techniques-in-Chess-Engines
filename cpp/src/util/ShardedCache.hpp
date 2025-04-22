#pragma once

#include "common.hpp"

template <typename KeyType, typename ValueType, size_t NumBuckets = 16> class ShardedCache {
public:
    ShardedCache() {
        // Initialize buckets.
        for (auto &bucket : m_buckets) {
            bucket.map.reserve(1024); // Preallocate space for 1024 elements.
        }
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

    bool contains(const KeyType &key) {
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

    void clear() {
        for (auto &bucket : m_buckets) {
            std::lock_guard lock(bucket.mutex); // Exclusive lock for writing.
            bucket.map.clear();
        }
    }

    size_t size() const {
        size_t totalSize = 0;
        for (const auto &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
            totalSize += bucket.map.size();
        }
        return totalSize;
    }

    // Check if the cache is empty.
    bool empty() const {
        for (const auto &bucket : m_buckets) {
            std::shared_lock lock(bucket.mutex); // Allow concurrent reads.
            if (!bucket.map.empty()) {
                return false;
            }
        }
        return true;
    }

public:
    // Add this to your ShardedCache class
public:
    // Iterator that locks all buckets during iteration
    class LockedIterator {
    private:
        ShardedCache &m_cache;
        std::vector<std::shared_lock<std::shared_mutex>> m_locks;
        using MapIterator = typename std::unordered_map<KeyType, ValueType>::iterator;
        size_t m_currentBucket = 0;
        MapIterator m_currentIt;
        MapIterator m_currentEnd;

    public:
        using value_type = std::pair<const KeyType, ValueType>;
        using reference = value_type &;
        using pointer = value_type *;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        LockedIterator(ShardedCache &c, bool isEnd = false) : m_cache(c) {
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

    LockedIterator begin() { return LockedIterator(*this); }
    LockedIterator end() { return LockedIterator(*this, true); }

private:
    struct Bucket {
        std::unordered_map<KeyType, ValueType> map;
        mutable std::shared_mutex mutex;
    };

    std::array<Bucket, NumBuckets> m_buckets;

    Bucket &getBucket(const KeyType &key) {
        // Simple hash to bucket index.
        size_t bucketIndex = std::hash<KeyType>{}(key) % NumBuckets;
        return m_buckets[bucketIndex];
    }
};
