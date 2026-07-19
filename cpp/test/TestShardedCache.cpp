#include "util/CollisionCheckedCache.hpp"
#include "util/ShardedCache.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void store(ShardedCache<int, int, 1> &cache, const int key, const int value) {
    require(cache.acquireOrInsert(key, -1), "new key was unexpectedly present");
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

    require(!cache.acquireOrInsert(1, -1), "existing key was not reused");
    int value = 0;
    require(cache.lookup(1, value) && value == 10, "cache hit returned the wrong value");
    cache.release(1);

    store(cache, 3, 30);
    require(cache.size() == 2, "cache exceeded its capacity");
    require(cache.evictionCount() == 1, "cache did not report its eviction");
    require(cache.acquireOrInsert(2, -1), "least-recently-used key was not evicted");
    cache.update(2, 20);
    cache.release(2);
}

void testLeasedEntriesAreNotEvicted() {
    ShardedCache<int, int, 1> cache(1);
    require(cache.acquireOrInsert(1, -1), "first lease did not insert");
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
    require(cache.acquireOrInsert(1, -1), "producer lease did not insert");
    require(!cache.acquireOrInsert(1, -1), "consumer lease inserted a duplicate");
    cache.update(1, 10);

    int value = 0;
    require(cache.lookup(1, value) && value == 10, "first consumer could not read");
    cache.release(1);
    require(cache.lookup(1, value) && value == 10, "second consumer lease was lost");
    cache.release(1);
}

void testStatisticsAndClear() {
    ShardedCache<int, std::vector<int>, 2> cache(4);
    require(cache.maximumSize() == 4, "maximum size was not retained");
    require(cache.acquireOrInsert(1, {}), "value insertion failed");
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

void testFingerprintCollisionRequiresExactIdentity() {
    CollisionCheckedCache<int, std::vector<int>, int, 1> cache(2);
    const std::vector<int> firstIdentity{1, 2, 3};
    const std::vector<int> collidingIdentity{4, 5, 6};

    require(cache.acquireOrInsert(7, firstIdentity, -1) == CacheAcquisition::Inserted,
            "first fingerprint did not insert");
    cache.update(7, firstIdentity, 42);
    cache.release(7);

    require(cache.acquireOrInsert(7, firstIdentity, -1) == CacheAcquisition::Hit,
            "exact identity was not a hit");
    int cachedValue = 0;
    require(cache.lookup(7, firstIdentity, cachedValue) && cachedValue == 42,
            "exact identity returned the wrong value");
    cache.release(7);

    require(cache.acquireOrInsert(7, collidingIdentity, -1) ==
                CacheAcquisition::FingerprintCollision,
            "colliding fingerprint was accepted as an exact hit");

    require(cache.acquireOrInsert(7, firstIdentity, -1) == CacheAcquisition::Hit,
            "collision handling damaged the original entry");
    cache.release(7);
}

} // namespace

int main() {
    testLeastRecentlyUsedEviction();
    testLeasedEntriesAreNotEvicted();
    testMultipleConsumersHoldIndependentLeases();
    testStatisticsAndClear();
    testFingerprintCollisionRequiresExactIdentity();
    return 0;
}
