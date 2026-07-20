#include "util/CollisionCheckedCache.hpp"
#include "util/ShardedCache.hpp"

#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void store(ShardedCache<int, int, 1> &cache, const int key, const int value) {
    const ShardedCache<int, int, 1>::Acquisition acquisition = cache.acquireOrInsert(key, -1);
    require(acquisition.inserted, "new key was unexpectedly present");
    cache.update(key, value);
    int storedValue = 0;
    require(cache.lookup(key, storedValue), "leased key was not found");
    require(storedValue == value, "stored value did not match");
    cache.release(key);
}

void testLeastRecentlyUsedEviction() {
    ShardedCache<int, int, 1> cache(2);
    store(cache, 1, 10);
    store(cache, 2, 20);

    const auto firstAcquisition = cache.acquireOrInsert(1, -1);
    require(!firstAcquisition.inserted, "existing key was not reused");
    require(*firstAcquisition.value == 10, "acquisition returned the wrong stored value");
    cache.release(1);

    store(cache, 3, 30);
    require(cache.size() == 2, "cache exceeded its capacity");
    require(cache.evictionCount() == 1, "cache did not report its eviction");
    const auto secondAcquisition = cache.acquireOrInsert(2, -1);
    require(secondAcquisition.inserted, "least-recently-used key was not evicted");
    cache.update(2, 20);
    cache.release(2);
}

void testLeasedEntriesAreNotEvicted() {
    ShardedCache<int, int, 1> cache(1);
    const auto firstAcquisition = cache.acquireOrInsert(1, -1);
    require(firstAcquisition.inserted, "first lease did not insert");
    cache.update(1, 10);

    store(cache, 2, 20);
    require(cache.size() == 1, "eviction did not restore the capacity");

    int value = 0;
    require(cache.lookup(1, value) && value == 10, "leased entry was evicted");
    cache.release(1);
    require(cache.size() == 1, "released entry exceeded capacity");
}

void testMultipleConsumersHoldIndependentLeases() {
    ShardedCache<int, int, 1> cache(1);
    const auto producerAcquisition = cache.acquireOrInsert(1, -1);
    require(producerAcquisition.inserted, "producer lease did not insert");
    const auto consumerAcquisition = cache.acquireOrInsert(1, -1);
    require(!consumerAcquisition.inserted, "consumer lease inserted a duplicate");
    cache.update(1, 10);

    require(*producerAcquisition.value == 10, "first consumer could not read");
    cache.release(1);
    require(*consumerAcquisition.value == 10, "second consumer lease was lost");
    cache.release(1);
}

void testStatisticsAndClear() {
    ShardedCache<int, std::vector<int>, 2> cache(4);
    require(cache.maximumSize() == 4, "maximum size was not retained");
    const auto acquisition = cache.acquireOrInsert(1, {});
    require(acquisition.inserted, "value insertion failed");
    cache.update(1, {1, 2, 3});
    cache.release(1);

    std::size_t valueCount = 0;
    cache.forEachValue([&valueCount](const std::vector<int> &value) {
        require(value.size() == 3, "visitor observed the wrong value");
        ++valueCount;
    });
    require(valueCount == 1, "visitor did not see every value");
    require(cache.estimatedStaticSizeBytes() > 0, "memory estimate was empty");

    cache.clear();
    require(cache.empty(), "clear did not remove cache entries");
}

void testProducerWakesConsumersWithSharedImmutableValue() {
    using Cache = CollisionCheckedCache<int, std::vector<int>, std::vector<int>, 1>;
    Cache cache(2);
    const std::vector<int> identity{1, 2, 3};

    Cache::Acquisition producer = cache.acquireOrInsert(7, identity);
    require(producer.status == CacheAcquisition::Inserted, "first request did not insert");
    require(producer.producer.has_value(), "inserter did not receive publish capability");

    Cache::Acquisition firstConsumer = cache.acquireOrInsert(7, identity);
    Cache::Acquisition secondConsumer = cache.acquireOrInsert(7, identity);
    require(firstConsumer.status == CacheAcquisition::Hit,
            "first consumer did not reuse the pending entry");
    require(secondConsumer.status == CacheAcquisition::Hit,
            "second consumer did not reuse the pending entry");
    require(!firstConsumer.producer.has_value() && !secondConsumer.producer.has_value(),
            "consumer received publish capability");

    std::future<const std::vector<int> *> waitingFirstConsumer =
        std::async(std::launch::async, [value = firstConsumer.value] { return &value.get(); });
    std::future<const std::vector<int> *> waitingSecondConsumer =
        std::async(std::launch::async, [value = secondConsumer.value] { return &value.get(); });
    require(waitingFirstConsumer.wait_for(10ms) == std::future_status::timeout &&
                waitingSecondConsumer.wait_for(10ms) == std::future_status::timeout,
            "consumers did not block for the producer");

    producer.producer->publish({4, 5, 6});
    require(waitingFirstConsumer.wait_for(1s) == std::future_status::ready &&
                waitingSecondConsumer.wait_for(1s) == std::future_status::ready,
            "consumers were not woken by publication");

    const std::vector<int> *firstValue = waitingFirstConsumer.get();
    const std::vector<int> *secondValue = waitingSecondConsumer.get();
    const std::vector<int> *producerValue = &producer.value.get();
    require(firstValue == producerValue && secondValue == producerValue,
            "consumers did not share the immutable stored value");
    require(*producerValue == std::vector<int>({4, 5, 6}), "published value was corrupted");

    cache.release(7);
    cache.release(7);
    cache.release(7);
}

void testFingerprintCollisionRemainsUncached() {
    using Cache = CollisionCheckedCache<int, std::vector<int>, int, 1>;
    Cache cache(2);
    const std::vector<int> firstIdentity{1, 2, 3};
    const std::vector<int> collidingIdentity{4, 5, 6};

    Cache::Acquisition inserted = cache.acquireOrInsert(7, firstIdentity);
    require(inserted.status == CacheAcquisition::Inserted, "first fingerprint did not insert");
    inserted.producer->publish(42);
    cache.release(7);

    Cache::Acquisition hit = cache.acquireOrInsert(7, firstIdentity);
    require(hit.status == CacheAcquisition::Hit, "exact identity was not a hit");
    require(hit.value.get() == 42, "exact identity returned the wrong value");
    cache.release(7);

    Cache::Acquisition collision = cache.acquireOrInsert(7, collidingIdentity);
    require(collision.status == CacheAcquisition::FingerprintCollision,
            "colliding fingerprint was accepted as an exact hit");
    require(!collision.value.valid() && !collision.producer.has_value(),
            "collision retained cache result state");

    Cache::Acquisition original = cache.acquireOrInsert(7, firstIdentity);
    require(original.status == CacheAcquisition::Hit,
            "collision handling damaged the original entry");
    cache.release(7);
}

void testLeasesProtectPendingSharedResultsFromEviction() {
    using Cache = CollisionCheckedCache<int, int, int, 1>;
    Cache cache(1);

    Cache::Acquisition first = cache.acquireOrInsert(1, 1);
    first.producer->publish(10);

    Cache::Acquisition second = cache.acquireOrInsert(2, 2);
    second.producer->publish(20);
    cache.release(2);

    require(cache.size() == 1, "unleased entry was not evicted at capacity");
    require(cache.evictionCount() == 1, "shared-result eviction was not recorded");
    require(first.value.get() == 10, "leased shared result was evicted");
    cache.release(1);
}

void testBrokenProducerWakesConsumersWithException() {
    using Cache = CollisionCheckedCache<int, int, int, 1>;
    Cache cache(1);
    Cache::ValueHandle value;
    {
        Cache::Acquisition acquisition = cache.acquireOrInsert(1, 1);
        value = acquisition.value;
        require(acquisition.producer.has_value(), "inserter did not receive producer");
    }

    require(value.wait_for(1s) == std::future_status::ready,
            "broken producer left consumers blocked");
    bool receivedBrokenPromise = false;
    try {
        static_cast<void>(value.get());
    } catch (const std::future_error &error) {
        receivedBrokenPromise =
            error.code() == std::make_error_code(std::future_errc::broken_promise);
    }
    require(receivedBrokenPromise, "consumer did not receive broken-promise failure");
    cache.release(1);
}

} // namespace

int main() {
    testLeastRecentlyUsedEviction();
    testLeasedEntriesAreNotEvicted();
    testMultipleConsumersHoldIndependentLeases();
    testStatisticsAndClear();
    testProducerWakesConsumersWithSharedImmutableValue();
    testFingerprintCollisionRemainsUncached();
    testLeasesProtectPendingSharedResultsFromEviction();
    testBrokenProducerWakesConsumersWithException();
    return 0;
}
