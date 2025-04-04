#include "Time.hpp"
#include "Log.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tensorboard_logger.h>

// Global timing results and a mutex to protect them.
std::unordered_map<std::string, double> function_times;
std::unordered_map<std::string, double> global_function_times;
std::unordered_map<std::string, int> global_function_invocations;
std::mutex time_mutex;

// Global start time (set on the first timed function invocation).
std::chrono::high_resolution_clock::time_point start_timing_time;
bool start_timing_time_initialized = false;

// Thread-local call stack to keep track of nested calls.
thread_local std::vector<std::string> call_stack;

TimeItGuard::TimeItGuard(const std::string &name)
    : func_name(name), start(std::chrono::high_resolution_clock::now()) {
    // Set global start time on first invocation.
    if (!start_timing_time_initialized) {
        start_timing_time = start;
        start_timing_time_initialized = true;
    }
    // Push the current function name onto the call stack.
    call_stack.push_back(func_name);
}

TimeItGuard::~TimeItGuard() {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::lock_guard<std::mutex> lock(time_mutex);
    // Update the accumulated time and invocation count for this function.
    function_times[func_name] += elapsed;
    global_function_invocations[func_name] += 1;

    // Pop the call stack (we assume matching push/pop).
    if (!call_stack.empty())
        call_stack.pop_back();

    // If we are inside a parent function, subtract the elapsed time from its total.
    if (!call_stack.empty()) {
        std::string parent = call_stack.back();
        function_times[parent] -= elapsed;
    }
}

void reset_times(TensorBoardLogger *logger, int iteration) {
    std::lock_guard<std::mutex> lock(time_mutex);

    // Sum the times recorded in function_times.
    double total_time = 0.0;
    for (const auto &[key, value] : function_times) {
        total_time += value;
    }

    // Add the current function times into global_function_times.
    for (auto &[key, value] : function_times) {
        global_function_times[key] += value;
    }

    // Sum the total global function times.
    double global_total_time = 0.0;
    for (const auto &[key, value] : global_function_times) {
        global_total_time += value;
    }

    if (total_time > 0) {
        // Prepare a sorted vector of (function name, accumulated time) pairs.
        std::vector<std::pair<std::string, double>> sorted;
        for (const auto &[key, value] : global_function_times) {
            sorted.emplace_back(key, value);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // Log scalar metrics for each function.
        for (const auto &[key, totalFuncTime] : sorted) {
            double percent = totalFuncTime / global_total_time * 100.0;
            logger->add_scalar(std::string("function_percent_of_execution_time/") + key, iteration,
                               percent);
            logger->add_scalar(std::string("function_total_time/") + key, iteration, totalFuncTime);
            logger->add_scalar(std::string("function_total_invocations/") + key, iteration,
                               static_cast<double>(global_function_invocations[key]));

            // Calculate local percentage for the function, if it exists in function_times.
            double local_percent =
                function_times.count(key) ? function_times[key] / total_time * 100.0 : 0.0;
            log(local_percent, "% (total", percent, "% on", global_function_invocations[key],
                "invocations)", key);
        }

        double total_elapsed = std::chrono::duration<double>(
                                   std::chrono::high_resolution_clock::now() - start_timing_time)
                                   .count();
        logger->add_scalar("function_time_total_traced_percent", iteration,
                           global_total_time / total_elapsed * 100.0);
        log("In total:", global_total_time / total_elapsed, "recorded");
    }

    // Clear the temporary function timing data.
    function_times.clear();
}