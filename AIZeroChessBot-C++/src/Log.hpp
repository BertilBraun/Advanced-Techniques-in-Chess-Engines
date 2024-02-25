#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Helper function to get the current time as a string
std::string currentTime() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::ctime(&time);
    auto str = ss.str();
    str.pop_back(); // Remove newline character
    return str;
}

// Convert various types to string
template <typename T> std::string toString(const T &value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

// Specialization for pairs
template <typename T1, typename T2> std::string toString(const std::pair<T1, T2> &pair) {
    return "(" + toString(pair.first) + ", " + toString(pair.second) + ")";
}

// Specialization for vectors
template <typename T> std::string toString(const std::vector<T> &vec) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += toString(vec[i]);
        if (i + 1 < vec.size())
            result += ", ";
    }
    result += "]";
    return result;
}

// Specialization for maps
template <typename K, typename V> std::string toString(const std::map<K, V> &map) {
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

// Variadic template log function
template <typename... Args> void log(Args... args) {
    logNoNewline(args...);
    logNoNewline("\n");
}

// Variadic template log function
template <typename... Args> void logNoNewline(Args... args) {
    std::ostringstream logStream;
    logStream << currentTime(), "- ";
    (logStream << ... << (toString(args) + " "));

    if constexpr (TO_FILE) {
        std::ofstream logFile("log.txt", std::ios::app);
        logFile << logStream.str();
        logFile.flush();
    } else {
        log(logStream.str();
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
