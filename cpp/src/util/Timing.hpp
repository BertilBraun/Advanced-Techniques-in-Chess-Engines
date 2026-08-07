#pragma once

#include <chrono>
#include <cstdint>

// Provides compact monotonic timing for search telemetry without exposing clock arithmetic.
class Stopwatch {
public:
    [[nodiscard]] std::uint64_t elapsedNanoseconds() const noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - m_startedAt)
                .count());
    }

    [[nodiscard]] std::int64_t elapsedMilliseconds() const noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - m_startedAt)
            .count();
    }

private:
    using Clock = std::chrono::steady_clock;
    const Clock::time_point m_startedAt = Clock::now();
};

class ScopedNanosecondTimer {
public:
    explicit ScopedNanosecondTimer(std::uint64_t &accumulator) noexcept
        : m_accumulator(accumulator) {}

    ~ScopedNanosecondTimer() { m_accumulator += m_stopwatch.elapsedNanoseconds(); }

    ScopedNanosecondTimer(const ScopedNanosecondTimer &) = delete;
    ScopedNanosecondTimer &operator=(const ScopedNanosecondTimer &) = delete;

private:
    std::uint64_t &m_accumulator;
    Stopwatch m_stopwatch;
};
