#pragma once

#include "common.hpp"

class Barrier {
    // This barrier implementation is not the most efficient, but it is simple and works
    // It waits for all threads to reach the barrier before continuing
    // It returns true for only one of the threads, which can be used to execute a block of code
    //      only on one of the threads

public:
    Barrier() = default;
    Barrier(size_t totalThreads) : m_totalThreads(totalThreads) {}

    Barrier &operator=(const Barrier &other) {
        m_barrierMutex.lock();
        m_barrierCount = other.m_barrierCount;
        m_totalThreads = other.m_totalThreads;
        m_barrierMutex.unlock();
        return *this;
    }

    bool barrier() {
        // return true for only one of the threads

        m_barrierMutex.lock();
        m_barrierCount++;
        m_barrierMutex.unlock();

        while (m_barrierCount < m_totalThreads) {
            std::this_thread::yield();
        }

        bool isFirst = false;
        m_barrierMutex.lock();
        if (m_barrierCount == m_totalThreads) {
            isFirst = true;
        }
        m_barrierCount = 0;
        m_barrierMutex.unlock();

        return isFirst;
    }

private:
    std::mutex m_barrierMutex;
    size_t m_barrierCount = 0;
    size_t m_totalThreads = 0;
};
