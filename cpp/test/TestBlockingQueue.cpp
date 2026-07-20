#include "util/BlockingQueue.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

struct TimedValue {
    int value;
    std::chrono::steady_clock::time_point enqueuedAt;
};

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testBoundedBackpressureAndCloseDrain() {
    BlockingQueue<int> queue(1);
    require(queue.push(1), "initial bounded push failed");

    std::future<bool> blockedPush =
        std::async(std::launch::async, [&queue] { return queue.push(2); });
    require(blockedPush.wait_for(20ms) == std::future_status::timeout,
            "bounded producer was not backpressured");

    require(queue.pop() == 1, "bounded queue returned the wrong first value");
    require(blockedPush.wait_for(1s) == std::future_status::ready && blockedPush.get(),
            "bounded producer did not resume");

    queue.close();
    require(queue.pop() == 2, "close discarded a queued value");
    require(!queue.pop().has_value(), "closed drained queue did not signal end-of-stream");
    require(!queue.push(3), "closed queue accepted a new value");
}

void testCloseReleasesBlockedProducer() {
    BlockingQueue<int> queue(1);
    require(queue.push(1), "initial close test push failed");
    std::future<bool> blockedPush =
        std::async(std::launch::async, [&queue] { return queue.push(2); });
    require(blockedPush.wait_for(20ms) == std::future_status::timeout,
            "close test producer was not blocked");

    queue.close();
    require(blockedPush.wait_for(1s) == std::future_status::ready && !blockedPush.get(),
            "close did not release the blocked producer");
    require(queue.pop() == 1, "close lost the value published before closure");
    require(!queue.pop().has_value(), "close test queue did not drain");
}

void testBoundedBulkBackpressureAndCapacityRejection() {
    BlockingQueue<int> queue(3);
    require(queue.push(1), "bounded bulk setup push failed");
    std::future<bool> blockedBulk =
        std::async(std::launch::async, [&queue] { return queue.pushBulk({2, 3, 4}); });
    require(blockedBulk.wait_for(20ms) == std::future_status::timeout,
            "bounded bulk publication bypassed capacity");

    require(queue.pop() == 1, "bounded bulk setup value was lost");
    require(blockedBulk.wait_for(1s) == std::future_status::ready && blockedBulk.get(),
            "bounded bulk publication did not resume atomically");
    queue.close();
    require(queue.pop() == 2 && queue.pop() == 3 && queue.pop() == 4,
            "bounded bulk publication changed ordering");

    BlockingQueue<int> smallerQueue(2);
    bool rejectedOversizedBulk = false;
    try {
        static_cast<void>(smallerQueue.pushBulk({1, 2, 3}));
    } catch (const std::invalid_argument &) {
        rejectedOversizedBulk = true;
    }
    require(rejectedOversizedBulk, "oversized bulk publication was not rejected");
    smallerQueue.close();
    require(!smallerQueue.pop().has_value(), "rejected bulk publication changed the queue");
}

void testMoveOnlyValues() {
    BlockingQueue<std::unique_ptr<int>> queue(3);
    std::vector<std::unique_ptr<int>> values;
    values.push_back(std::make_unique<int>(1));
    values.push_back(std::make_unique<int>(2));
    require(queue.pushBulk(std::move(values)), "move-only bulk publication failed");
    require(queue.push(std::make_unique<int>(3)), "move-only single publication failed");
    queue.close();

    std::optional<std::unique_ptr<int>> first = queue.pop();
    std::optional<std::unique_ptr<int>> second = queue.pop();
    std::optional<std::unique_ptr<int>> third = queue.pop();
    require(first.has_value() && **first == 1, "first move-only value was corrupted");
    require(second.has_value() && **second == 2, "second move-only value was corrupted");
    require(third.has_value() && **third == 3, "third move-only value was corrupted");
    require(!queue.pop().has_value(), "move-only queue did not drain");
}

void testBulkPublishAndCloseDrain() {
    BlockingQueue<int> queue;
    std::future<std::optional<std::vector<int>>> waitingBatch =
        std::async(std::launch::async, [&queue] {
            return queue.popBatch(
                3, [](const int &) { return std::chrono::steady_clock::now() + 1s; });
        });

    require(waitingBatch.wait_for(20ms) == std::future_status::timeout,
            "empty queue did not block its consumer");
    require(queue.pushBulk({1, 2, 3}), "bulk publication failed");
    require(waitingBatch.wait_for(1s) == std::future_status::ready,
            "bulk publication did not wake the consumer");
    require(waitingBatch.get() == std::vector<int>({1, 2, 3}),
            "bulk publication changed value ordering");

    require(queue.pushBulk({4, 5}), "second bulk publication failed");
    queue.close();
    require(queue.pop() == 4 && queue.pop() == 5, "close did not drain bulk values");
    require(!queue.pop().has_value(), "bulk queue did not reach end-of-stream");
}

void testBulkPublishWakesEveryConsumer() {
    BlockingQueue<int> queue;
    std::future<std::optional<int>> firstConsumer =
        std::async(std::launch::async, [&queue] { return queue.pop(); });
    std::future<std::optional<int>> secondConsumer =
        std::async(std::launch::async, [&queue] { return queue.pop(); });
    require(firstConsumer.wait_for(20ms) == std::future_status::timeout &&
                secondConsumer.wait_for(20ms) == std::future_status::timeout,
            "bulk consumers did not begin waiting");

    require(queue.pushBulk({1, 2}), "multi-consumer bulk publication failed");
    require(firstConsumer.wait_for(1s) == std::future_status::ready &&
                secondConsumer.wait_for(1s) == std::future_status::ready,
            "bulk publication left a consumer stranded");
    const int first = *firstConsumer.get();
    const int second = *secondConsumer.get();
    require((first == 1 && second == 2) || (first == 2 && second == 1),
            "bulk consumers received the wrong values");
    queue.close();
}

void testBatchMaximumReturnsImmediately() {
    BlockingQueue<TimedValue> queue;
    const auto enqueuedAt = std::chrono::steady_clock::now();
    require(queue.pushBulk({{1, enqueuedAt}, {2, enqueuedAt}, {3, enqueuedAt}}),
            "timed values were not published");

    const auto startedAt = std::chrono::steady_clock::now();
    const std::optional<std::vector<TimedValue>> batch =
        queue.popBatch(2, [](const TimedValue &value) { return value.enqueuedAt + 1s; });
    const auto elapsed = std::chrono::steady_clock::now() - startedAt;

    require(batch.has_value() && batch->size() == 2, "maximum-sized batch was not returned");
    require((*batch)[0].value == 1 && (*batch)[1].value == 2,
            "maximum-sized batch changed ordering");
    require(elapsed < 200ms, "maximum-sized batch waited for its deadline");
    queue.close();
    require(queue.pop()->value == 3, "batch maximum discarded the backlog");
}

void testOldestItemControlsBatchDeadline() {
    BlockingQueue<TimedValue> queue;
    const auto firstEnqueuedAt = std::chrono::steady_clock::now();
    require(queue.push(TimedValue{1, firstEnqueuedAt}), "first timed value was not published");

    const auto startedAt = std::chrono::steady_clock::now();
    std::future<std::optional<std::vector<TimedValue>>> waitingBatch =
        std::async(std::launch::async, [&queue] {
            return queue.popBatch(3,
                                  [](const TimedValue &value) { return value.enqueuedAt + 80ms; });
        });

    std::this_thread::sleep_for(30ms);
    require(queue.push(TimedValue{2, firstEnqueuedAt + 2s}),
            "second timed value was not published");

    require(waitingBatch.wait_for(1s) == std::future_status::ready,
            "deadline batch did not finish");
    const auto elapsed = std::chrono::steady_clock::now() - startedAt;
    const std::optional<std::vector<TimedValue>> batch = waitingBatch.get();
    require(batch.has_value() && batch->size() == 2, "deadline batch had the wrong size");
    require((*batch)[0].value == 1 && (*batch)[1].value == 2, "deadline batch changed ordering");
    require(elapsed >= 60ms && elapsed < 500ms,
            "newer item reset or bypassed the oldest item's deadline");
    queue.close();
}

void testCloseInterruptsBatchDeadlineAndDrains() {
    BlockingQueue<TimedValue> queue;
    require(queue.push(TimedValue{1, std::chrono::steady_clock::now()}),
            "close deadline value was not published");
    std::future<std::optional<std::vector<TimedValue>>> waitingBatch =
        std::async(std::launch::async, [&queue] {
            return queue.popBatch(3, [](const TimedValue &value) { return value.enqueuedAt + 5s; });
        });
    require(waitingBatch.wait_for(20ms) == std::future_status::timeout,
            "deadline batch did not begin waiting");

    queue.close();
    require(waitingBatch.wait_for(1s) == std::future_status::ready,
            "close did not interrupt the batch deadline");
    const std::optional<std::vector<TimedValue>> batch = waitingBatch.get();
    require(batch.has_value() && batch->size() == 1 && batch->front().value == 1,
            "close did not drain the waiting batch");
    require(!queue.pop().has_value(), "closed deadline queue did not reach end-of-stream");
}

} // namespace

int main() {
    testBoundedBackpressureAndCloseDrain();
    testCloseReleasesBlockedProducer();
    testBoundedBulkBackpressureAndCapacityRejection();
    testMoveOnlyValues();
    testBulkPublishAndCloseDrain();
    testBulkPublishWakesEveryConsumer();
    testBatchMaximumReturnsImmediately();
    testOldestItemControlsBatchDeadline();
    testCloseInterruptsBatchDeadlineAndDrains();
    return 0;
}
