#pragma once

#include "Log.hpp"
#include <cmath>
#include <vector>

inline std::map<std::string, long long> __timeit_results;

// Time a function and add the result to the timeit results
// Should be callable like this:
// timeit([&] { return someFunction(); }, "someFunction");
template <typename Func> auto timeit(Func func, const std::string &funcName) {
    using ReturnType = decltype(func()); // Deduce the return type of the function

    if constexpr (std::is_same_v<ReturnType, void>) {
        auto start = std::chrono::high_resolution_clock::now();

        // If the function returns void
        func(); // Just call the function

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        __timeit_results[funcName] += duration;
    } else {
        auto start = std::chrono::high_resolution_clock::now();

        // If the function returns a value
        auto result = func(); // Call the function and store its result

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        __timeit_results[funcName] += duration;

        return result; // Return the result of the function
    }
}

inline std::string getTimeitResults() {
    PrettyTable table({"Function", "Accumulated Time (s)"});

    for (auto &pair : __timeit_results) {
        table.addRow(pair.first, (double) pair.second * 1e-9);
    }

    return table.getAsString();
}
