#pragma once

#include "common.hpp"

#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

class Subprocess {

public:
    Subprocess(const std::string &command, const std::string &mode) {
        pipe = popen(command.c_str(), mode.c_str());
        if (!pipe)
            throw std::runtime_error("Failed to start subprocess.");
    }

    ~Subprocess() { pclose(pipe); }

    // Overload the insertion operator for writing to the subprocess
    Subprocess &operator<<(const std::string &input) {
        fputs(input.c_str(), pipe);
        fputc('\n', pipe);
        fflush(pipe);
        return *this;
    }

    // Overload the extraction operator for reading from the subprocess
    Subprocess &operator>>(std::string &output) {
        std::array<char, 1024> buffer;
        output.clear();
        if (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            output = buffer.data();
        }
        return *this;
    }

    std::vector<std::string> readLines() {
        std::vector<std::string> lines;
        std::string line;
        (*this) >> line;
        while (!line.empty()) {
            lines.push_back(line);
            (*this) >> line;
        }
        return lines;
    }

private:
    FILE *pipe = nullptr;
};
