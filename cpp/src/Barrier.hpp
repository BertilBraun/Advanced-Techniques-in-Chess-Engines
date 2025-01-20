#pragma once

#include "common.hpp"

static inline constexpr size_t MAX_NUM_BARRIERS = 100;

class Barrier {
    // This barrier implementation is not the most efficient, but it is simple and works
    // It waits for all threads to reach the barrier before continuing
    // It returns true for only one of the threads, which can be used to execute a block of code
    //      only on one of the threads

public:
    Barrier() = default;
    Barrier(int totalThreads) : m_totalThreads(totalThreads) {}

    bool barrier() {
        // return true for only one of the threads

        m_barrierMutex.lock();
        int myCallCount = m_barrierCallCount % MAX_NUM_BARRIERS;

        if (m_barrierCounts[myCallCount] >= m_totalThreads)
            myCallCount = ++m_barrierCallCount % MAX_NUM_BARRIERS;

        m_barrierCounts[myCallCount]++;
        m_barrierMutex.unlock();

        while (m_barrierCounts[myCallCount] < m_totalThreads)
            std::this_thread::yield();

        bool isFirst = false;
        m_barrierMutex.lock();
        if (m_barrierCounts[myCallCount] == m_totalThreads)
            isFirst = true;

        m_barrierCounts[myCallCount]++;
        if (m_barrierCounts[myCallCount] >= 2 * m_totalThreads)
            m_barrierCounts[myCallCount] = 0;

        m_barrierMutex.unlock();

        return isFirst;
    }

    void updateRequired(int delta) {
        m_barrierMutex.lock();
        m_totalThreads += delta;
        m_barrierMutex.unlock();
    }

private:
    std::mutex m_barrierMutex;
    std::array<std::atomic<int>, MAX_NUM_BARRIERS> m_barrierCounts{0};
    size_t m_barrierCallCount = 0;
    int m_totalThreads = 0;
};
