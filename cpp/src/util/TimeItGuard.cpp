#include "TimeItGuard.h"
#include "Log.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <ranges>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>


// Global timing results and a mutex to protect them.
std::unordered_map<std::string, float> functionTimes;
std::unordered_map<std::string, float> globalFunctionTimes;
std::unordered_map<std::string, int> globalFunctionInvocations;
std::mutex timeMutex;

// Global start time (set on the first timed function invocation).
std::chrono::high_resolution_clock::time_point startTimingTime;
bool startTimingTimeInitialized = false;

// Thread-local call stack to keep track of nested calls.
thread_local std::vector<std::string> callStack;

TimeItGuard::TimeItGuard(const std::string &name)
    : m_funcName(name), m_start(std::chrono::high_resolution_clock::now()) {
    // Set global start time on first invocation.
    if (!startTimingTimeInitialized) {
        startTimingTime = m_start;
        startTimingTimeInitialized = true;
    }
    // Push the current function name onto the call stack.
    callStack.push_back(m_funcName);
}

TimeItGuard::~TimeItGuard() {
    const float elapsedSeconds = elapsed();

    std::lock_guard<std::mutex> lock(timeMutex);
    // Update the accumulated time and invocation count for this function.
    functionTimes[m_funcName] += elapsedSeconds;
    globalFunctionInvocations[m_funcName] += 1;

    // Pop the call stack (we assume matching push/pop).
    if (!callStack.empty())
        callStack.pop_back();

    // If we are inside a parent function, subtract the elapsed time from its total.
    if (!callStack.empty()) {
        const std::string parent = callStack.back();
        functionTimes[parent] -= elapsedSeconds;
    }
}

void resetTimes() {
    std::lock_guard<std::mutex> lock(timeMutex);

    // Sum the times recorded in function_times.
    float totalTime = 0.0;
    for (const auto &value : functionTimes | std::views::values) {
        totalTime += value;
    }

    // Add the current function times into global_function_times.
    for (auto &[key, value] : functionTimes) {
        globalFunctionTimes[key] += value;
    }

    // Sum the total global function times.
    float globalTotalTime = 0.0;
    for (const auto &value : globalFunctionTimes | std::views::values) {
        globalTotalTime += value;
    }

    if (totalTime > 0) {
        // Prepare a sorted vector of (function name, accumulated time) pairs.
        std::vector<std::pair<std::string, float>> sorted;
        for (const auto &[key, value] : globalFunctionTimes) {
            sorted.emplace_back(key, value);
        }
        std::ranges::sort(sorted,
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // Log scalar metrics for each function.
        for (const auto &[key, totalFuncTime] : sorted) {
            const float percent = totalFuncTime / globalTotalTime * 100.0;

            // Calculate local percentage for the function, if it exists in function_times.
            const float localPercent =
                functionTimes.contains(key) ? functionTimes[key] / totalTime * 100.0 : 0.0;
            log(localPercent, "% (total", percent, "% on", globalFunctionInvocations[key],
                "invocations)", key);
        }

        const float totalElapsed = std::chrono::duration<float>(
                                 std::chrono::high_resolution_clock::now() - startTimingTime)
                                 .count();

        log("In total:", globalTotalTime / totalElapsed * 100.f, "% recorded");
    }

    // Clear the temporary function timing data.
    functionTimes.clear();
}

float TimeItGuard::elapsed() const {
    // Calculate the elapsed time in seconds.
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start);
    return duration.count() / 1000.0f;
}