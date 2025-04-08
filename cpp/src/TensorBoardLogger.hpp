#pragma once

#include <string>
#include <vector>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "util/json.hpp" // Include the nlohmann/json library

class TensorBoardLogger {
public:
    TensorBoardLogger(const std::string &log_dir) : log_dir(log_dir), stop_thread(false) {
        // Start periodic flush thread.
        flush_thread = std::thread(&TensorBoardLogger::periodic_flush, this);
    }

    ~TensorBoardLogger() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop_thread = true;
        }
        if (flush_thread.joinable()) {
            flush_thread.join();
        }
        flush();
    }

    template <typename T> void add_scalar(const std::string &tag, int step, T value) {
        nlohmann::json j;
        j["type"] = "scalar";
        j["tag"] = tag;
        j["step"] = step;
        j["value"] = value;
        std::lock_guard<std::mutex> lock(mu);
        events.push_back(j);
    }

    template <typename T>
    void add_histogram(const std::string &tag, int step, const std::vector<T> &values) {
        nlohmann::json j;
        j["type"] = "histogram";
        j["tag"] = tag;
        j["step"] = step;
        j["values"] = values;
        std::lock_guard<std::mutex> lock(mu);
        events.push_back(j);
    }

    void add_text(const std::string &tag, int step, const std::string &text) {
        nlohmann::json j;
        j["type"] = "text";
        j["tag"] = tag;
        j["step"] = step;
        j["text"] = text;
        std::lock_guard<std::mutex> lock(mu);
        events.push_back(j);
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mu);
        if (events.empty())
            return;

        // Create JSON array from events.
        nlohmann::json j = events;

        // Generate a new file path.
        std::string file_path = getNewFilePath();
        std::ofstream ofs(file_path);
        if (ofs) {
            ofs << j.dump(4);
            // Clear events after flushing.
            events.clear();
        }
    }

    std::string getNewFilePath() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time);
        std::ostringstream oss;
        oss << log_dir << "/events_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
        return oss.str();
    }

    void periodic_flush() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mu);
                if (stop_thread)
                    break;
                flush();
                std::this_thread::sleep_for(std::chrono::minutes(1));
            }
        }
    }

    std::string log_dir;
    std::mutex mu;
    std::vector<nlohmann::json> events;
    std::thread flush_thread;
    bool stop_thread;
};