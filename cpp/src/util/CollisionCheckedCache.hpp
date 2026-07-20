#pragma once

#include "ShardedCache.hpp"

#include <cstddef>
#include <exception>
#include <future>
#include <optional>
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
    using ValueHandle = std::shared_future<ValueType>;

    class Producer {
    public:
        Producer(const Producer &) = delete;
        Producer &operator=(const Producer &) = delete;
        Producer(Producer &&) noexcept = default;
        Producer &operator=(Producer &&) noexcept = default;

        void publish(ValueType value) { m_promise.set_value(std::move(value)); }
        void publishException(std::exception_ptr exception) {
            m_promise.set_exception(std::move(exception));
        }

    private:
        friend class CollisionCheckedCache;

        explicit Producer(std::promise<ValueType> promise) : m_promise(std::move(promise)) {}

        std::promise<ValueType> m_promise;
    };

    struct Acquisition {
        CacheAcquisition status;
        ValueHandle value;
        std::optional<Producer> producer;
    };

    explicit CollisionCheckedCache(const std::size_t maximumSize) : m_cache(maximumSize) {}

    [[nodiscard]] Acquisition acquireOrInsert(const FingerprintType &fingerprint,
                                              const IdentityType &identity) {
        std::optional<Producer> producer;
        const typename Cache::Acquisition cacheAcquisition =
            m_cache.acquireOrInsertWithFactory(fingerprint, [&identity, &producer] {
                std::promise<ValueType> promise;
                ValueHandle value = promise.get_future().share();
                producer.emplace(Producer(std::move(promise)));
                return Entry{identity, std::move(value)};
            });
        if (cacheAcquisition.inserted) {
            return {CacheAcquisition::Inserted, cacheAcquisition.value->value, std::move(producer)};
        }

        const Entry &cachedEntry = *cacheAcquisition.value;
        if (cachedEntry.identity == identity) {
            return {CacheAcquisition::Hit, cachedEntry.value, std::nullopt};
        }

        m_cache.release(fingerprint);
        return {CacheAcquisition::FingerprintCollision, {}, std::nullopt};
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
        ValueHandle value;
    };

    using Cache = ShardedCache<FingerprintType, Entry, NumBuckets, Hash>;
    Cache m_cache;
};
