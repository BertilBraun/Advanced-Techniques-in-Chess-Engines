#pragma once

#include <chrono>
#include <string>

class TensorBoardLogger;

class TimeItGuard {
public:
    // On entry, record the function name and start time.
    explicit TimeItGuard(std::string name);

    // On exit, calculate the elapsed time and update globals.
    ~TimeItGuard();

    [[nodiscard]] float elapsed() const;

private:
    std::string m_funcName;
    std::chrono::high_resolution_clock::time_point m_start;
};

struct FunctionTimeInfo {
    std::string name; // Name of the function
    float percent; // Percentage of total time spent in this function
    float total;   // Total time spent in this function
    int invocations; // Number of times this function was invoked
};

struct TimeInfo {
    float totalTime;
    float percentRecorded;
    std::vector<FunctionTimeInfo> functionTimes;
};


TimeInfo resetTimes();

// #define ENABLE_TIMING
#ifdef ENABLE_TIMING
#define TIMEIT(name) TimeItGuard timer(name);
#else
#define TIMEIT(name)
#endif