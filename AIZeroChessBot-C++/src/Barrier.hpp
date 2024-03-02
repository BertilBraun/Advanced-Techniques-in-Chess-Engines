#pragma once

#include "common.hpp"

class Barrier {
    // This barrier implementation is not the most efficient, but it is simple and works
    // It waits for all threads to reach the barrier before continuing
    // It returns true for only one of the threads, which can be used to execute a block of code
    //      only on one of the threads

public:
    Barrier() = default;
    Barrier(size_t totalThreads) : m_totalThreads(totalThreads) { m_barrierCounts.insert({0, 0}); }

    Barrier &operator=(const Barrier &other) {
        m_barrierMutex.lock();
        m_barrierCallCount = other.m_barrierCallCount;
        m_totalThreads = other.m_totalThreads;
        m_barrierMutex.unlock();
        return *this;
    }

    bool barrier() {
        // return true for only one of the threads

        m_barrierMutex.lock();
        if (m_barrierCounts[m_barrierCallCount] == m_totalThreads) {
            m_barrierCounts.insert({++m_barrierCallCount, 0});
        }
        int myCallCount = m_barrierCallCount;
        m_barrierCounts[myCallCount]++;
        m_barrierMutex.unlock();

        while (m_barrierCounts[myCallCount] < m_totalThreads) {
            std::this_thread::yield();
        }

        bool isFirst = false;
        m_barrierMutex.lock();
        if (m_barrierCounts[myCallCount] == m_totalThreads) {
            isFirst = true;
        }
        m_barrierCounts[myCallCount]++;
        if (m_barrierCounts[myCallCount] == 2 * m_totalThreads) {
            m_barrierCounts.erase(myCallCount);
        }
        m_barrierMutex.unlock();

        return isFirst;
    }

    void updateRequired(size_t delta) {
        m_barrierMutex.lock();
        m_totalThreads += delta;
        m_barrierMutex.unlock();
    }

private:
    std::mutex m_barrierMutex;
    std::unordered_map<size_t, size_t> m_barrierCounts;
    size_t m_barrierCallCount = 0;
    size_t m_totalThreads = 0;
};
