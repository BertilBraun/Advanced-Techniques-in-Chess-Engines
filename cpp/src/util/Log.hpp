#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

std::string currentTime();

template <typename T> std::string toString(const T &value) {
    // Convert various types to string
    std::stringstream ss;
    ss << value;
    return ss.str();
}

namespace chess {
class Move;
}

std::string toString(const chess::Move &move);

template <typename T1, typename T2> std::string toString(const std::pair<T1, T2> &pair) {
    // Specialization for pairs
    return "(" + toString(pair.first) + ", " + toString(pair.second) + ")";
}

template <typename T> std::string toString(const std::vector<T> &vec) {
    // Specialization for vectors
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += toString(vec[i]);
        if (i + 1 < vec.size())
            result += ", ";
    }
    result += "]";
    return result;
}

template <typename K, typename V> std::string toString(const std::map<K, V> &map) {
    // Specialization for maps
    std::string result = "{";
    for (auto it = map.begin(); it != map.end(); ++it) {
        result += toString(*it);
        if (std::next(it) != map.end())
            result += ", ";
    }
    result += "}";
    return result;
}

constexpr bool TO_FILE = true;
constexpr bool TO_STDERR = true;

static inline std::map<std::thread::id, std::string> threadIds;

template <bool AddNewline, typename... Args> void logCommon(Args... args) {
    // Common log function that writes to file and/or stderr

    std::thread::id threadId = std::this_thread::get_id();
    if (threadIds.find(threadId) == threadIds.end()) {
        // Start IDs from 1 for readability
        threadIds[threadId] = toString(threadIds.size() + 1);
        if (threadIds[threadId].size() == 1)
            threadIds[threadId] = "0" + threadIds[threadId];
    }

    std::ostringstream logStream;
    logStream << '[' << currentTime() << "] [" << threadIds[threadId] << "] ";

    auto appendToLog = [&logStream](const auto &arg) {
        std::string argStr = toString(arg);
        logStream << argStr;
        // Check if the last character is not a newline before adding a space
        if (!argStr.empty() && argStr.back() != '\n') {
            logStream << ' ';
        }
    };

    (appendToLog(args), ...);

    if constexpr (AddNewline) {
        logStream << std::endl;
    }

    std::string logString = logStream.str();
    if (logString.back() == ' ')
        logString.pop_back(); // Remove trailing space

    if constexpr (TO_FILE) {
        std::string logPath = "log_" + toString(threadIds[threadId]) + ".txt";
        std::ofstream logFile(logPath, std::ios::app);
        logFile << logString;
        logFile.flush();
    }
    if constexpr (TO_STDERR) {
        std::cerr << logString;
        std::cerr.flush();
    }
}

// Public log functions that wrap the common function
template <typename... Args> void log(Args... args) { logCommon<true>(args...); }

template <typename... Args> void logNoNewline(Args... args) { logCommon<false>(args...); }

// Progress bar function
// Usage:
// for (auto _ : range(100)) {
//     tqdm(i, 100, "Processing");
//     ...
// }
bool tqdm(size_t current, size_t total, std::string desc = "", int width = 50);

class PrettyTable {
    // Helper class to print a table with a header and rows
    // Example usage:
    // PrettyTable table({"Column 1", "Column 2", "Column 3"});
    // table.addRow(1, 2, 3);
    // table.addRow(4, 5, 6);
    // log(table.toString());
public:
    PrettyTable(const std::vector<std::string> &columns) : m_columns(columns) {}

    template <typename... Args> void addRow(Args... args) {
        if (sizeof...(args) != m_columns.size()) {
            throw std::runtime_error("Row size does not match column size");
        }
        m_rows.push_back({toString(args)...});
    }

    std::string getAsString() const;

private:
    std::vector<std::string> m_columns;
    std::vector<std::vector<std::string>> m_rows;
};