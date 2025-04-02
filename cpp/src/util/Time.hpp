#pragma once

#include <chrono>
#include <string>

class TensorBoardLogger;

class TimeItGuard {
public:
    // On entry, record the function name and start time.
    TimeItGuard(const std::string &name);

    // On exit, calculate the elapsed time and update globals.
    ~TimeItGuard();

private:
    std::string func_name;
    std::chrono::high_resolution_clock::time_point start;
};

void reset_times(TensorBoardLogger *logger, int iteration);
