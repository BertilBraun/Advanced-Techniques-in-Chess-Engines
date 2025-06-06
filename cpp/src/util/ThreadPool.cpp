#include "ThreadPool.hpp"

ThreadPool::ThreadPool(unsigned int nThreads)
{
    for (unsigned int i = 0; i < nThreads; ++i)
        m_workers.emplace_back([this]{ workerLoop(); });
}

ThreadPool::~ThreadPool()
{
    shutdown();
}

void ThreadPool::workerLoop()
{
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock lk(m_mutex);
            m_cv.wait(lk, [this]{ return m_stopping || !m_tasks.empty(); });
            if (m_stopping && m_tasks.empty())
                return;
            job = std::move(m_tasks.front());
            m_tasks.pop();
        }
        job();
    }
}

void ThreadPool::shutdown()
{
    {
        std::lock_guard lk(m_mutex);
        if (m_stopping) return;
        m_stopping = true;
    }
    m_cv.notify_all();
    for (std::thread& t : m_workers) t.join();
}
