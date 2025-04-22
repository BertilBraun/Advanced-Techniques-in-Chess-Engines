#include "Log.hpp"

#include "chess.hpp"

std::string toString(const chess::Move &move) {
    // Specialization for Move type
    return move.uci();
}

std::string currentTime() {
    // Helper function to get the current time as a string
    // Format: "hh:mm:ss:ms"
    auto now = std::chrono::system_clock::now();
    auto inTimeT = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&inTimeT), "%X");
    return ss.str();
}

// Global map to keep track of start times for each unique progress bar.
std::unordered_map<std::string, std::chrono::steady_clock::time_point> startTimes;

std::string formatDuration(std::chrono::steady_clock::duration duration) {
    auto hrs = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hrs;
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= mins;
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);

    std::ostringstream oss;
    if (hrs.count() > 0)
        oss << hrs.count() << ":";
    if (mins.count() > 0)
        oss << std::setfill('0') << std::setw(hrs.count() > 0 ? 2 : 0) << mins.count() << ":";
    oss << std::setfill('0') << std::setw(2) << secs.count();
    return oss.str();
}

bool tqdm(size_t current, size_t total, std::string desc, int width) {
    if (startTimes.find(desc) == startTimes.end()) {
        // If this is the first time this desc is being used, set the start time.
        startTimes[desc] = std::chrono::steady_clock::now();
    }

    float progress = std::min((float) current / (float) total, 1.0f);
    int pos = (int) ((float) width * progress);

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

    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - startTimes[desc];
    auto eta =
        std::chrono::duration_cast<std::chrono::seconds>(elapsed / progress * (1 - progress));
    auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    auto elementsPerSecond = (elapsedSeconds > 0) ? (float) current / (float) elapsedSeconds : 0.0f;

    bar << int(progress * 100.0) << "% ";

    if (current < total) {
        bar << current << "/" << total << " ";

        bar << "[" << formatDuration(elapsed) << "<" << formatDuration(eta) << ", " << std::fixed
            << std::setprecision(2) << elementsPerSecond << "it/s] ";

        bar << desc << "\r";
    } else {
        bar << total << "/" << total << " ";

        bar << "[" << formatDuration(elapsed) << ", " << std::fixed << std::setprecision(2)
            << elementsPerSecond << "it/s] ";

        bar << desc << "\n";
    }

    logNoNewline(bar.str());

    // If we've reached 100%, remove the start time for this desc.
    if (current >= total) {
        startTimes.erase(desc);
    }

    return current < total;
}

std::string PrettyTable::getAsString() const {
    std::vector<size_t> columnWidths(m_columns.size(), 0);
    for (size_t i = 0; i < m_columns.size(); ++i) {
        columnWidths[i] = std::max(columnWidths[i], m_columns[i].size());
    }
    for (const auto &row : m_rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            columnWidths[i] = std::max(columnWidths[i], row[i].size());
        }
    }

    std::string result;
    auto addRowToResult = [&result, &columnWidths](const std::vector<std::string> &row) {
        result += "| ";
        for (size_t i = 0; i < row.size(); ++i) {
            result += row[i];
            result += std::string(columnWidths[i] - row[i].size(), ' ');
            result += " | ";
        }
        result += "\n";
    };

    addRowToResult(m_columns);
    result += std::string((m_columns.size() + 1) * 3 +
                              std::accumulate(columnWidths.begin(), columnWidths.end(), 0),
                          '-') +
              "\n";
    for (const auto &row : m_rows) {
        addRowToResult(row);
    }
    return result;
}