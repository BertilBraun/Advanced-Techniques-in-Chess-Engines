#pragma once

#include <chrono>
#include <string>

class TensorBoardLogger;

class TimeItGuard {
public:
    // On entry, record the function name and start time.
    explicit TimeItGuard(const std::string &name);

    // On exit, calculate the elapsed time and update globals.
    ~TimeItGuard();

    float elapsed() const;

private:
    std::string m_funcName;
    std::chrono::high_resolution_clock::time_point m_start;
};

void resetTimes();