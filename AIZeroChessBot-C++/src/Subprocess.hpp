#pragma once

#include "common.hpp"

#include <cstdio>
#include <istream>
#include <stdexcept>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

class Subprocess {

public:
    Subprocess(const std::string &command) {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        auto threadId = oss.str();

        auto logFileName = "subprocess_" + threadId + ".log";

        pipe = popen((command + " > " + logFileName + " 2>&1").c_str(), "w");
        if (!pipe)
            throw std::runtime_error("Failed to start subprocess.");

        // clear the log file
        fclose(fopen(logFileName.c_str(), "w"));

        logFile = fopen(logFileName.c_str(), "r");
        if (!logFile)
            throw std::runtime_error("Failed to open log file.");
    }

    ~Subprocess() {
        pclose(pipe);
        fclose(logFile);
    }

    // Overload the insertion operator for writing to the subprocess
    Subprocess &operator<<(const std::string &input) {
        fputs(input.c_str(), pipe);
        fputc('\n', pipe);
        fflush(pipe);
        return *this;
    }

    // Overload the extraction operator for reading one line from the subprocess log file
    Subprocess &operator>>(std::string &output) {
        output.clear();

        char c;
        while (fread(&c, 1, 1, logFile) == 1) {
            if (c == '\n')
                break;
            output.push_back(c);
        }

        return *this;
    }

private:
    FILE *pipe = nullptr;
    FILE *logFile = nullptr;
};
