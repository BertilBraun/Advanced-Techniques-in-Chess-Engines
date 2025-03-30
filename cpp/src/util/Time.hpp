#pragma once

#include "Log.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>


// Global timing results and mutex.
inline std::unordered_map<std::string, std::atomic<long long>> __timeit_results;
inline std::mutex __timeit_mutex;

template <typename Func> auto timeit(Func func, const std::string &funcName) {
    using ReturnType = decltype(func());
    auto start = std::chrono::high_resolution_clock::now();

    // Lambda to update timing results for the given function name.
    auto update = [&](long long duration) {
        if (auto it = __timeit_results.find(funcName); it != __timeit_results.end()) {
            it->second.fetch_add(duration, std::memory_order_relaxed);
        } else {
            std::lock_guard<std::mutex> lock(__timeit_mutex);
            if (auto it = __timeit_results.find(funcName); it != __timeit_results.end()) {
                it->second.fetch_add(duration, std::memory_order_relaxed);
            } else {
                __timeit_results.emplace(funcName, duration);
            }
        }
    };

    if constexpr (std::is_same_v<ReturnType, void>) {
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        update(duration);
    } else {
        auto result = func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        update(duration);
        return result;
    }
}

inline std::string getTimeitResults() {
    PrettyTable table({"Function", "Accumulated Time (s)"});

    for (auto &pair : __timeit_results) {
        table.addRow(pair.first, (double) pair.second * 1e-9);
    }

    return table.getAsString();
}
