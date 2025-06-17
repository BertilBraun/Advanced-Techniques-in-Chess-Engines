#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(unsigned int nThreads);
    ~ThreadPool();

    /// Enqueue an arbitrary callable and get a future to its result.
    template <class F, class... Args>
    [[nodiscard]] auto enqueue(F &&f, Args &&...args)
        -> std::future<std::invoke_result_t<F, Args...>>;

    /// Gracefully stop accepting work and join all threads.
    void shutdown();

    [[nodiscard]] unsigned int numThreads() const {
        return static_cast<unsigned int>(m_workers.size());
    }

private:
    void workerLoop();

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stopping = false;
};

// ----- template method in the header so the compiler can see it ----------
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>> {
    using Ret = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<Ret()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<Ret> res = task->get_future();

    {
        std::lock_guard lk(m_mutex);
        if (m_stopping)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        m_tasks.emplace([task] { (*task)(); });
    }
    m_cv.notify_one();
    return res;
}
