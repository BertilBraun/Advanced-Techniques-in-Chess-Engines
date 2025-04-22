#pragma once

#include <string>
#include <vector>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "util/json.hpp" // Include the nlohmann/json library

class TensorBoardLogger {
public:
    TensorBoardLogger(const std::string &log_dir) : m_logDir(log_dir), m_stopThread(false) {
        // Start periodic flush thread.
        m_flushThread = std::thread(&TensorBoardLogger::periodicFlush, this);
    }

    ~TensorBoardLogger() {
        std::cout << "Stopping TensorBoard logger..." << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mu);
            m_stopThread = true;
        }
        if (m_flushThread.joinable()) {
            m_flushThread.join();
        }
        flush();
    }

    template <typename T> void addScalar(const std::string &tag, int step, T value) {
        nlohmann::json j;
        j["type"] = "scalar";
        j["tag"] = tag;
        j["step"] = step;
        j["value"] = value;
        pushEvent(j);
    }

    template <typename T>
    void addHistogram(const std::string &tag, int step, const std::vector<T> &values) {
        nlohmann::json j;
        j["type"] = "histogram";
        j["tag"] = tag;
        j["step"] = step;
        j["values"] = values;
        pushEvent(j);
    }

    void addText(const std::string &tag, int step, const std::string &text) {
        nlohmann::json j;
        j["type"] = "text";
        j["tag"] = tag;
        j["step"] = step;
        j["text"] = text;
        pushEvent(j);
    }

private:
    void pushEvent(const nlohmann::json &event) {
        std::lock_guard<std::mutex> lock(m_mu);
        m_events.push_back(event);
        if (m_events.size() >= 100) {
            std::cout << "WARNING: TensorBoard logger buffer is full. Flushing to disk manually."
                      << std::endl;
            flush();
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(m_mu);
        if (m_events.empty())
            return;

        // Create JSON array from events.
        nlohmann::json j = m_events;

        // Generate a new file path.
        std::string filePath = getNewFilePath();
        std::ofstream ofs(filePath);
        if (ofs) {
            ofs << j.dump(4);
            // Clear events after flushing.
            m_events.clear();
        }
    }

    std::string getNewFilePath() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time);
        std::ostringstream oss;
        oss << m_logDir << "/events_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".json";
        return oss.str();
    }

    void periodicFlush() {
        while (true) {
            if (m_stopThread)
                break;
            flush();
            std::this_thread::sleep_for(std::chrono::minutes(1));
        }
    }

    std::string m_logDir;
    std::mutex m_mu;
    std::vector<nlohmann::json> m_events;
    std::thread m_flushThread;
    bool m_stopThread;
};