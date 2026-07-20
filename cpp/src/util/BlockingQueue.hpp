#pragma once

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename Value> class BlockingQueue {
public:
    BlockingQueue() = default;

    explicit BlockingQueue(const std::size_t capacity) : m_capacity(capacity) {
        assert(capacity > 0);
    }

    BlockingQueue(const BlockingQueue &) = delete;
    BlockingQueue &operator=(const BlockingQueue &) = delete;

    [[nodiscard]] bool push(Value value) {
        {
            std::unique_lock lock(m_mutex);
            m_notFull.wait(lock, [this] {
                return m_closed || !m_capacity.has_value() || m_values.size() < *m_capacity;
            });
            if (m_closed) {
                return false;
            }
            m_values.push(std::move(value));
        }
        m_notEmpty.notify_one();
        return true;
    }

    [[nodiscard]] bool pushBulk(std::vector<Value> values) {
        if (m_capacity.has_value() && values.size() > *m_capacity) {
            throw std::invalid_argument("bulk publication exceeds queue capacity");
        }

        bool published = false;
        {
            std::unique_lock lock(m_mutex);
            m_notFull.wait(lock, [this, &values] {
                return m_closed || !m_capacity.has_value() ||
                       values.size() <= *m_capacity - m_values.size();
            });
            if (m_closed) {
                return false;
            }
            for (Value &value : values) {
                m_values.push(std::move(value));
                published = true;
            }
        }
        if (published) {
            m_notEmpty.notify_all();
        }
        return true;
    }

    [[nodiscard]] std::optional<Value> pop() {
        std::optional<Value> value;
        {
            std::unique_lock lock(m_mutex);
            m_notEmpty.wait(lock, [this] { return m_closed || !m_values.empty(); });
            if (m_values.empty()) {
                return std::nullopt;
            }
            value.emplace(std::move(m_values.front()));
            m_values.pop();
        }
        m_notFull.notify_one();
        return value;
    }

    template <typename FrontDeadline>
    [[nodiscard]] std::optional<std::vector<Value>> popBatch(const std::size_t maximumBatchSize,
                                                             FrontDeadline frontDeadline) {
        assert(maximumBatchSize > 0);
        std::vector<Value> values;
        {
            std::unique_lock lock(m_mutex);
            m_notEmpty.wait(lock, [this] { return m_closed || !m_values.empty(); });
            if (m_values.empty()) {
                return std::nullopt;
            }

            const auto deadline = frontDeadline(m_values.front());
            m_notEmpty.wait_until(lock, deadline, [this, maximumBatchSize] {
                return m_closed || m_values.size() >= maximumBatchSize;
            });

            const std::size_t batchSize = std::min(maximumBatchSize, m_values.size());
            values.reserve(batchSize);
            for (std::size_t i = 0; i < batchSize; ++i) {
                values.push_back(std::move(m_values.front()));
                m_values.pop();
            }
        }
        m_notFull.notify_all();
        return values;
    }

    void close() {
        {
            std::lock_guard lock(m_mutex);
            m_closed = true;
        }
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }

private:
    std::optional<std::size_t> m_capacity;
    std::mutex m_mutex;
    std::condition_variable m_notEmpty;
    std::condition_variable m_notFull;
    std::queue<Value> m_values;
    bool m_closed = false;
};
