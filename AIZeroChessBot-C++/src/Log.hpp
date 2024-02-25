#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

std::string currentTime() {
    // Helper function to get the current time as a string
    // Format: "hh:mm:ss:ms"
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%X");
    return ss.str();
}

template <typename T> std::string toString(const T &value) {
    // Convert various types to string
    std::stringstream ss;
    ss << value;
    return ss.str();
}

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

static inline std::map<std::thread::id, size_t> __THREAD_IDS;

template <bool AddNewline, typename... Args> void logCommon(Args... args) {
    // Common log function that writes to file and/or stderr

    std::thread::id threadId = std::this_thread::get_id();
    if (__THREAD_IDS.find(threadId) == __THREAD_IDS.end()) {
        __THREAD_IDS[threadId] = __THREAD_IDS.size() + 1; // Start IDs from 1 for readability
    }

    std::ostringstream logStream;
    logStream << '[' << currentTime() << "] [ " << __THREAD_IDS[threadId] << " ] ";
    (logStream << ... << (toString(args) + " "));
    if constexpr (AddNewline) {
        logStream << std::endl;
    }

    if constexpr (TO_FILE) {
        std::string logPath = "log_" + toString(__THREAD_IDS[threadId]) + ".txt";
        std::ofstream logFile(logPath, std::ios::app);
        logFile << logStream.str();
        logFile.flush();
    }
    if constexpr (TO_STDERR) {
        std::cerr << logStream.str();
        std::cerr.flush();
    }
}

// Public log functions that wrap the common function
template <typename... Args> void log(Args... args) { logCommon<true>(args...); }

template <typename... Args> void logNoNewline(Args... args) { logCommon<false>(args...); }

inline bool tqdm(size_t current, size_t total, std::string desc = "", int width = 50) {
    float progress = std::min((float) current / total, 1.0f);
    int pos = (int) (width * progress);

    std::ostringstream bar;

    bar << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos)
            bar << "=";
        else if (i == pos)
            bar << ">";
        else
            bar << " ";
    }
    bar << "]";

    logNoNewline(bar.str(), int(progress * 100.0), '%', desc, (current == total ? "\n" : "\r"));
    return current < total;
}
