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

template <typename... Args> void log(Args... args) {
    // Variadic template log function
    std::ostringstream logStream;
    logStream << '[' << currentTime() << "] ";
    (logStream << ... << (toString(args) + " "));
    logStream << std::endl;

    if constexpr (TO_FILE) {
        std::ofstream logFile("log.txt", std::ios::app);
        logFile << logStream.str();
        logFile.flush();
    }
    if constexpr (TO_STDERR) {
        std::cerr << logStream.str();
        std::cerr.flush();
    }
}

template <typename... Args> void logNoNewline(Args... args) {
    // Variadic template log function without a newline
    std::ostringstream logStream;
    logStream << '[' << currentTime() << "] ";
    (logStream << ... << (toString(args) + " "));

    if constexpr (TO_FILE) {
        std::ofstream logFile("log.txt", std::ios::app);
        logFile << logStream.str();
        logFile.flush();
    }
    if constexpr (TO_STDERR) {
        std::cerr << logStream.str();
        std::cerr.flush();
    }
}

inline bool tqdm(size_t current, size_t total, std::string desc = "", int width = 50) {
    float progress = std::min((float) current / total, 1.0f);
    int pos = (int) (width * progress);

    logNoNewline("[");
    for (int i = 0; i < width; ++i) {
        if (i < pos)
            logNoNewline("=");
        else if (i == pos)
            logNoNewline(">");
        else
            logNoNewline(" ");
    }
    logNoNewline(']', int(progress * 100.0), '%', desc, '\r');
    if (current == total) {
        logNoNewline("\n");
    }
    return current < total;
}
