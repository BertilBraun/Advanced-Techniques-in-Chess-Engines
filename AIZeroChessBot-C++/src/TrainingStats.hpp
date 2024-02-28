#pragma once

#include "common.hpp"

class TrainingStats {
public:
    TrainingStats() = default;

    void update(float policyLoss, float valueLoss, float totalLoss) {
        m_policyLoss += policyLoss;
        m_valueLoss += valueLoss;
        m_totalLoss += totalLoss;
        ++m_numBatches;
    }

    // Overload the + operator to add two TrainingStats objects
    TrainingStats operator+(const TrainingStats &other) const {
        return TrainingStats(m_policyLoss + other.m_policyLoss, m_valueLoss + other.m_valueLoss,
                             m_totalLoss + other.m_totalLoss, m_numBatches + other.m_numBatches);
    }
    TrainingStats &operator+=(const TrainingStats &other) {
        *this = *this + other;
        return *this;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Policy Loss: " << m_policyLoss / (float) m_numBatches
               << ", Value Loss: " << m_valueLoss / (float) m_numBatches
               << ", Total Loss: " << m_totalLoss / (float) m_numBatches << ", Batches: " << m_numBatches;
        return stream.str();
    }

private:
    float m_policyLoss = 0.0f;
    float m_valueLoss = 0.0f;
    float m_totalLoss = 0.0f;
    int m_numBatches = 0;

    TrainingStats(float policyLoss, float valueLoss, float totalLoss, int numBatches)
        : m_policyLoss(policyLoss), m_valueLoss(valueLoss), m_totalLoss(totalLoss),
          m_numBatches(numBatches) {}
};

class LearningStats {
public:
    LearningStats() = default;

    void update(size_t numMoves, const TrainingStats &stats) {
        m_totalNumMoves += numMoves;
        m_trainingStats.push_back(stats);
        ++m_totalIterations;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Total Moves: " << m_totalNumMoves << "\n";
        stream << "Total Iterations: " << m_totalIterations << "\n";
        for (size_t i = 0; i < m_trainingStats.size(); ++i) {
            stream << "Iteration " << i + 1 << ": " << m_trainingStats[i].toString() << "\n";
        }
        return stream.str();
    }

private:
    size_t m_totalNumMoves = 0;
    size_t m_totalIterations = 0;
    std::vector<TrainingStats> m_trainingStats;
};

class SelfPlayStats {
public:
    SelfPlayStats() = default;

    void update(size_t numMoves, float result) {
        if (numMoves == 0) {
            log("Warning: Self play game with 0 moves");
        }

        ++m_totalNumGames;
        m_totalNumMoves += numMoves;
        if (result == -1.0f) {
            ++m_totalLosses;
        } else if (result == 1.0f) {
            ++m_totalWins;
        } else {
            ++m_totalDraws;
            m_totalDrawResult += result;
        }
    }

    std::string toString() const {
        float totalResult = m_totalDrawResult + (float) (m_totalWins - m_totalLosses);
        float averageResult = totalResult / (float) m_totalNumGames;

        std::ostringstream stream;
        stream << "Total Games: " << m_totalNumGames << "\n";
        stream << "Total Moves: " << m_totalNumMoves << "\n";
        stream << "Average Moves Per Game: " << (float) m_totalNumMoves / (float) m_totalNumGames
               << "\n";
        stream << "Total Result: " << totalResult << " (Wins - Losses + Draws)\n";
        stream << "Total Draw Result: " << m_totalDrawResult << "\n";
        stream << "Average Result: " << averageResult << " (1.0 = Win, 0.0 = Draw, -1.0 = Loss)\n";
        stream << "Total Draws: " << m_totalDraws << "\n";
        stream << "Total Wins: " << m_totalWins << "\n";
        stream << "Total Losses: " << m_totalLosses << "\n";
        return stream.str();
    }

    SelfPlayStats operator+(const SelfPlayStats &other) const {
        return SelfPlayStats(m_totalNumGames + other.m_totalNumGames,
                             m_totalNumMoves + other.m_totalNumMoves,
                             m_totalDraws + other.m_totalDraws, m_totalWins + other.m_totalWins,
                             m_totalLosses + other.m_totalLosses);
    }
    SelfPlayStats &operator+=(const SelfPlayStats &other) {
        *this = *this + other;
        return *this;
    }

private:
    size_t m_totalNumGames = 0;
    size_t m_totalNumMoves = 0;
    size_t m_totalDraws = 0;
    size_t m_totalWins = 0;
    size_t m_totalLosses = 0;
    float m_totalDrawResult = 0.0f;

    SelfPlayStats(size_t totalNumGames, size_t totalNumMoves, size_t totalDraws, size_t totalWins,
                  size_t totalLosses)
        : m_totalNumGames(totalNumGames), m_totalNumMoves(totalNumMoves), m_totalDraws(totalDraws),
          m_totalWins(totalWins), m_totalLosses(totalLosses) {}
};
