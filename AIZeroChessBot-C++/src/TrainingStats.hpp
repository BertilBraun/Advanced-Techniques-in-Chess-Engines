#pragma once

#include "common.hpp"

class TrainingStats {
public:
    float policyLoss = 0.0f;
    float valueLoss = 0.0f;
    float totalLoss = 0.0f;
    int numBatches = 0;

    void update(float policyLoss, float valueLoss, float totalLoss) {
        this->policyLoss += policyLoss;
        this->valueLoss += valueLoss;
        this->totalLoss += totalLoss;
        ++numBatches;
    }

    // Overload the + operator to add two TrainingStats objects
    TrainingStats operator+(const TrainingStats &other) const {
        return TrainingStats{policyLoss + other.policyLoss, valueLoss + other.valueLoss,
                             totalLoss + other.totalLoss, numBatches + other.numBatches};
    }
    TrainingStats &operator+=(const TrainingStats &other) {
        *this = *this + other;
        return *this;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Policy Loss: " << policyLoss / numBatches
               << ", Value Loss: " << valueLoss / numBatches
               << ", Total Loss: " << totalLoss / numBatches;
        return stream.str();
    }
};

class LearningStats {
public:
    size_t totalNumMoves = 0;
    size_t totalIterations = 0;
    std::vector<TrainingStats> trainingStats;

    void update(size_t numMoves, const TrainingStats &stats) {
        totalNumMoves += numMoves;
        trainingStats.push_back(stats);
        ++totalIterations;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Total Moves: " << totalNumMoves << "\n";
        stream << "Total Iterations: " << totalIterations << "\n";
        for (size_t i = 0; i < trainingStats.size(); ++i) {
            stream << "Iteration " << i + 1 << ": " << trainingStats[i].toString() << "\n";
        }
        return stream.str();
    }
};

class SelfPlayStats {
public:
    int totalNumGames = 0;
    int totalNumMoves = 0;
    int totalDraws = 0;
    int totalWins = 0;
    int totalLosses = 0;
    float totalDrawResult = 0.0f;

    void update(int numMoves, float result) {
        if (numMoves == 0) {
            log("Warning: Self play game with 0 moves");
        }

        ++totalNumGames;
        totalNumMoves += numMoves;
        if (result == -1.0f) {
            ++totalLosses;
        } else if (result == 1.0f) {
            ++totalWins;
        } else {
            ++totalDraws;
            totalDrawResult += result;
        }
    }

    std::string toString() const {
        float totalResult = totalDrawResult + totalWins - totalLosses;
        float averageResult = totalResult / (float) totalNumGames;

        std::ostringstream stream;
        stream << "Total Games: " << totalNumGames << "\n";
        stream << "Total Moves: " << totalNumMoves << "\n";
        stream << "Average Moves Per Game: " << (float) totalNumMoves / (float) totalNumGames
               << "\n";
        stream << "Total Result: " << totalResult << " (Wins - Losses + Draws)\n";
        stream << "Total Draw Result: " << totalDrawResult << "\n";
        stream << "Average Result: " << averageResult << " (1.0 = Win, 0.0 = Draw, -1.0 = Loss)\n";
        stream << "Total Draws: " << totalDraws << "\n";
        stream << "Total Wins: " << totalWins << "\n";
        stream << "Total Losses: " << totalLosses << "\n";
        return stream.str();
    }

    SelfPlayStats operator+(const SelfPlayStats &other) const {
        return SelfPlayStats{totalNumGames + other.totalNumGames,
                             totalNumMoves + other.totalNumMoves, totalDraws + other.totalDraws,
                             totalWins + other.totalWins, totalLosses + other.totalLosses};
    }
    SelfPlayStats &operator+=(const SelfPlayStats &other) {
        *this = *this + other;
        return *this;
    }
};
